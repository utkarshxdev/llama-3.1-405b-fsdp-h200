#!/usr/bin/env python3
"""
train.py — Production LoRA fine-tuning for Llama-3.1-405B on 8x H200 (1.13 TB VRAM).

Launch command:
    torchrun --nproc_per_node=8 train.py

Architecture & Design Decisions:
    - Pure bf16 (No 4-bit/8-bit quantization): With 8x H200 (1.13 TB total VRAM),
      the full 405B base model fits natively in bfloat16 sharded across 8 GPUs (~101 GB/GPU).
    - PyTorch FSDP (full_shard): Shards model parameters, gradients, and optimizer states
      across all 8 ranks with transformer-layer wrapping (LlamaDecoderLayer).
    - Scaled Dot-Product Attention (SDPA): Memory-efficient attention natively integrated.
    - NaNSafeCallback: Pre-optimizer callback that clamps/zeros NaN gradients on degenerate
      batches, preventing poisoned weights during long cluster runs.
    - Decoupled LoRA Checkpoint Resumption: Bypasses PyTorch FSDP optimizer state-sync
      deadlocks on restart by loading adapter weights directly via PeftModel.from_pretrained(),
      allowing instant, rank-safe resume.
"""

import os
import sys
import math
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import Dataset

# ─────────────────────────────────────────────────────────────
# Configuration (Environment-driven with clean defaults)
# ─────────────────────────────────────────────────────────────
MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Llama-3.1-405B-Instruct")
DATASET_PATH = os.environ.get("DATASET_PATH", "./data/training_dataset.jsonl")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./output/Llama-3.1-405B-LoRA-FSDP")
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))  # O(n^2) attention guard
TRAIN_SPLIT = float(os.environ.get("TRAIN_SPLIT", "0.9"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_405b")


# ─────────────────────────────────────────────────────────────
# NaN-Safe Trainer Callback
# ─────────────────────────────────────────────────────────────
class NaNSafeCallback(TrainerCallback):
    """Zero out NaN/Inf gradients to prevent run poisoning on degenerate batches."""

    def on_pre_optimizer_step(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        nan_count = 0
        for name, p in model.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                p.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                nan_count += 1
        if nan_count > 0:
            logger.warning(
                "Step %d: zeroed NaN gradients in %d parameters", state.global_step, nan_count
            )


# ─────────────────────────────────────────────────────────────
# Dataset Ingestion & Formatting
# ─────────────────────────────────────────────────────────────
def format_single_chat(messages):
    """Format message sequence into standard Llama-3.1 instruct chat template."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(
            f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"
        )
    return "<|begin_of_text|>" + "".join(parts)


def format_prompt(examples):
    """Batched formatting function for TRL SFTTrainer."""
    output_texts = []
    messages_list = examples.get("messages", [])
    if isinstance(messages_list, list) and len(messages_list) > 0 and isinstance(messages_list[0], list):
        for msgs in messages_list:
            output_texts.append(format_single_chat(msgs))
    elif isinstance(messages_list, list) and len(messages_list) > 0 and isinstance(messages_list[0], dict):
        output_texts.append(format_single_chat(messages_list))
    else:
        output_texts.append("")
    return output_texts


def load_training_dataset():
    """Load JSONL dataset from disk and construct train/eval splits."""
    path = Path(DATASET_PATH)
    if not path.exists():
        logger.error("Dataset not found at %s. Please set DATASET_PATH.", DATASET_PATH)
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    formatted = [{"text": format_single_chat(row["messages"])} for row in data if "messages" in row]
    ds = Dataset.from_list(formatted)
    ds = ds.shuffle(seed=42)
    split = ds.train_test_split(test_size=1.0 - TRAIN_SPLIT, seed=42)

    logger.info("Dataset loaded: %d train / %d eval samples", len(split["train"]), len(split["test"]))
    return split["train"], split["test"]


# ─────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────
def main():
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)

    logger.info("=" * 64)
    logger.info("Llama-3.1-405B FSDP LoRA Fine-Tuning (8x H200 SXM, Pure bf16)")
    logger.info("=" * 64)

    # 1. Tokenizer initialization
    logger.info("Initializing tokenizer: %s", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Base model loading (bf16, CPU memory offload prior to FSDP sharding)
    logger.info("Loading 405B base model in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False  # Mandatory for activation checkpointing

    # 3. Dataset preparation
    train_ds, eval_ds = load_training_dataset()

    # 4. Global schedule calculations
    num_train_samples = len(train_ds)
    effective_batch = 8 * 1 * 8  # 8 GPUs * batch_size=1 * grad_accum=8 = 64
    steps_per_epoch = math.ceil(num_train_samples / effective_batch)
    total_steps = steps_per_epoch * 3

    logger.info("Effective global batch size: %d", effective_batch)
    logger.info("Steps per epoch: %d | Total target steps: %d", steps_per_epoch, total_steps)

    # 5. Decoupled Checkpoint Resumption (Bypasses PyTorch FSDP multi-rank deadlocks)
    last_ckpt = get_last_checkpoint(OUTPUT_DIR)
    if last_ckpt is not None:
        logger.info("Discovered existing checkpoint: %s — executing decoupled adapter restore...", last_ckpt)
        # Load adapter weights directly without distributed optimizer state lockup
        model = PeftModel.from_pretrained(model, last_ckpt, is_trainable=True)
        lora_config = None           # Already wrapped
        warmup_steps = 0             # Weights already warm
        learning_rate = 2e-5         # Cold optimizer state; use stable conservative LR
        lr_scheduler_type = "constant"
    else:
        logger.info("No prior checkpoint found — initializing fresh LoRA adapters.")
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        warmup_steps = int(0.03 * total_steps)
        learning_rate = 2e-4
        lr_scheduler_type = "cosine"

    # 6. Distributed Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        # Micro-batching for 405B parameters at 2048 sequence length
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,

        # Scheduling
        max_steps=total_steps,
        warmup_steps=warmup_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        weight_decay=0.01,

        # Precision (Pure bfloat16)
        bf16=True,
        fp16=False,

        # Memory optimizations
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch",

        # Evaluation & Saving
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        load_best_model_at_end=False,   # Avoids distributed cross-rank checkpoint sync stalls

        # Monitoring
        logging_steps=5,
        logging_first_step=True,
        report_to="tensorboard",

        # CRITICAL: Always False — manual decoupled resumption handles checkpoint state above
        resume_from_checkpoint=False,

        # FSDP Sharding Configuration
        fsdp="full_shard auto_wrap",
        fsdp_config={
            "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
            "activation_checkpointing": True,
            "sync_module_states": True,
        },

        # Data loader performance
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
        max_grad_norm=1.0,
    )

    # 7. Trainer instantiation
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="text",
        callbacks=[NaNSafeCallback()],
    )

    # 8. Execute Training
    logger.info("Starting distributed training across 8 ranks...")
    trainer.train()

    # 9. Save final consolidated LoRA adapter
    final_adapter_dir = os.path.join(OUTPUT_DIR, "final_adapter")
    logger.info("Saving consolidated LoRA adapter to %s", final_adapter_dir)
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)

    logger.info("=" * 64)
    logger.info("Training complete. Artifacts successfully written to %s", final_adapter_dir)
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
