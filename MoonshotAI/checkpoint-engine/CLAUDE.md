# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation
```bash
# Basic installation
pip install checkpoint-engine

# With P2P support (includes mooncake-transfer-engine)
pip install 'checkpoint-engine[p2p]'
```

### Testing
```bash
# Run correctness test (requires 8 GPUs)
torchrun --nproc-per-node 8 tests/test_update.py
```

### Running Examples
```bash
# Basic weight update example (requires 8 GPUs)
torchrun --nproc-per-node 8 examples/update.py --update-method all --checkpoint-path /path/to/model

# Save global metadata for reuse by new instances
torchrun --nproc-per-node 8 examples/update.py --checkpoint-path /path/to/model \
    --sleep-time 300 --save-metas-file global_metas.pkl

# Load from existing metadata
torchrun --nproc-per-node 8 examples/update.py --load-metas-file global_metas.pkl
```

## Architecture

### Core Components

1. **ParameterServer** (checkpoint_engine/ps.py): The main weight update service with two implementations:
   - **Broadcast**: Fast synchronous weight updates for multiple inference instances
   - **P2P**: Asynchronous updates for dynamically added instances using mooncake-transfer-engine

2. **VllmColocateWorkerExtension** (checkpoint_engine/worker.py): vLLM worker extension for coordinating with the parameter server

3. **Weight Update Pipeline**: Three-stage process (H2D → Broadcast → Reload) with overlapped communication and copy for optimal performance

### Key Features

- Supports various data types: BF16, FP16, FP8, Float32
- Optimized for large models (tested up to 1T parameters)
- GPU memory-aware pipelining with automatic fallback to serial execution
- RDMA support for P2P transfers between nodes
- Integration with vLLM inference engine via ZeroMQ sockets

### vLLM Integration

Requires vLLM with `/collective_rpc` API endpoint. Start vLLM with:
```bash
VLLM_SERVER_DEV_MODE=1 python3 -m vllm.entrypoints.openai.api_server \
    --worker-extension-cls checkpoint_engine.worker.VllmColocateWorkerExtension \
    --load-format dummy --tensor-parallel-size=8
```

### FP8 Support

FP8 quantization requires applying the patch in `patches/vllm_fp8.patch` to vLLM. Only tested with DeepSeek-V3.1 and Kimi-K2 models.