# Llama-3.1-405B Production LoRA Fine-Tuning on 8x NVIDIA H200 (1.13 TB VRAM)

Production-grade PyTorch FSDP training harness for fine-tuning **Llama-3.1-405B-Instruct** across an 8× NVIDIA H200 SXM5 node in pure `bfloat16`.

---

## Key Architectural Highlights

1. **Pure bf16 (Zero Quantization Overhead):**
   - With 8× H200 GPUs (141 GB HBM3e each, 1.13 TB total VRAM), the full 405B parameter model fits natively in bfloat16 sharded across all ranks (~101 GB/GPU).
   - Eliminates dequantization latency and quantization errors introduced by 4-bit/8-bit wrappers.

2. **Decoupled LoRA Checkpoint Resumption (Bypassing FSDP Deadlocks):**
   - Standard PyTorch FSDP `resume_from_checkpoint=True` frequently deadlocks during distributed optimizer state agreement across ranks.
   - This script decouples LoRA adapter weights from the distributed optimizer buffer sync, reloading weights via `PeftModel.from_pretrained()` and reinitializing the optimizer with a conservative learning rate.

3. **`NaNSafeCallback`:**
   - Pre-optimizer callback that dynamically clamps and zeroes NaN/Inf gradients on degenerate batches, preventing gradient explosion from poisoning multi-day cluster runs.

4. **Scaled Dot-Product Attention (SDPA):**
   - Uses PyTorch's native SDPA attention backend for memory-efficient forward and backward passes.

---

## Launch Command

```bash
torchrun --nproc_per_node=8 train.py
```

### Environment Overrides

```bash
export MODEL_ID="meta-llama/Llama-3.1-405B-Instruct"
export DATASET_PATH="./data/training_dataset.jsonl"
export OUTPUT_DIR="./output/Llama-405B-LoRA-FSDP"
export MAX_SEQ_LENGTH=2048

torchrun --nproc_per_node=8 train.py
```

---

## Hardware & Distributed Configuration

- **Topology:** 8× NVIDIA H200 SXM5 (141 GB HBM3e per GPU, NVLink 900 GB/s bidirectional)
- **Parallelism Strategy:** PyTorch FSDP (`full_shard auto_wrap`)
- **Wrapped Layer:** `LlamaDecoderLayer`
- **Activation Checkpointing:** Enabled
- **Precision:** Pure `bfloat16`
