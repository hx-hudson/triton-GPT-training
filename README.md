# Triton GPT Training

A GPT training project with custom GPU kernels implemented in [Triton](https://triton-lang.org/).

This project replaces key PyTorch operators with custom Triton implementations, focusing on **LayerNorm** and **FlashAttention**, and integrates them into the GPT training pipeline with custom autograd support.

## Implemented Triton Kernels

### LayerNorm

Implemented custom LayerNorm forward and backward kernels in Triton.

Key techniques:

- Row-wise mean and variance reduction
- FP32 accumulation for numerical stability
- Custom backward computation for:
  - input gradient (`dx`)
  - weight gradient (`dw`)
  - bias gradient (`db`)
- Intermediate gradient reduction for `dw` and `db`
- Atomic synchronization for concurrent partial-gradient accumulation
- Separate reduction kernel for final `dw` / `db`
- Triton autotuning for:
  - block size
  - number of warps
  - reduction tile sizes

### FlashAttention

Implemented causal FlashAttention forward and backward passes in Triton.

Key techniques:

- Block-wise tiled attention computation
- Causal masking
- Online softmax using running maximum and normalization sum
- Log-sum-exp (`LSE`) caching for backward
- Fused attention computation without materializing the full attention matrix
- Custom backward computation for:
  - `dQ`
  - `dK`
  - `dV`
- FP32 accumulation for numerically sensitive operations
- BF16 Tensor Core-friendly `tl.dot` paths
- Separate macro/micro tiling strategy in the backward pass
- Triton autotuning for:
  - query/output tile size
  - key/value tile size
  - backward macro/micro tile sizes
  - number of warps
  - pipeline stages

## Mixed Precision

The training pipeline supports BF16 mixed-precision execution with PyTorch autocast.

The Triton kernels keep numerically sensitive calculations such as softmax statistics and reductions in FP32 while using BF16 inputs for matrix multiplication where appropriate.

This allows the kernels to benefit from Tensor Core execution while preserving numerical stability.

## Autotuning

Instead of using fixed compile-time constants, the Triton kernels use `@triton.autotune` to benchmark multiple kernel configurations and select the fastest configuration for the current tensor shape.

Parameters tuned include:

- `BLOCK_SIZE`
- `BLOCK_SIZE_QO`
- `BLOCK_SIZE_KV`
- `BLOCK_SIZE_MACRO`
- `BLOCK_SIZE_MICRO`
- `BLOCK_SIZE_M`
- `BLOCK_SIZE_N`
- `num_warps`
- `num_stages`

This makes the kernels less dependent on manually selected tile sizes and allows them to adapt better to different workloads.

## Benchmark

Benchmark configuration:

- Tokens per step: **8,192**
- Mixed precision: **BF16**
- Workload: GPT training step including forward, backward, and optimizer update

| Implementation | Step Time | Tokens/s | Peak GPU Memory |
|---|---:|---:|---:|
| PyTorch baseline | 104.012 ms | 78,760 | 6.496 GB |
| Triton kernels + autotune | **101.537 ms** | **80,680** | 6.565 GB |

Compared with the PyTorch baseline, the Triton implementation achieved approximately:

- **2.44% lower step time**
- **2.44% higher training throughput**

The result shows that custom Triton kernels can match and slightly outperform highly optimized PyTorch operators while providing direct control over tiling, memory access, precision, and kernel scheduling.

## Project Goals

The goal of this project is not only to reproduce existing operators, but to understand GPU kernel optimization at a lower level, including:

- memory access patterns
- tiling
- GPU occupancy
- register and shared-memory pressure
- Tensor Core utilization
- numerical stability
- kernel fusion
- forward/backward kernel design
- workload-specific autotuning

## Notes

The exact performance depends on GPU architecture, tensor shapes, precision, Triton version, and PyTorch version. Autotuning results are hardware- and workload-dependent.
