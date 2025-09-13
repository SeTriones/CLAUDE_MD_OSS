# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Installation and Setup
```bash
# Install SGLang with all dependencies
pip install -e "python/[all]"

# Install for specific hardware configurations
pip install -e "python/[all]"        # NVIDIA GPU
pip install -e "python/[all_hip]"    # AMD GPU  
pip install -e "python/[all_cpu]"    # CPU
pip install -e "python/[all_npu]"    # Ascend NPU

# Install development dependencies
pip install -e "python/[dev]"
```

### Building and Testing
```bash
# Format code (requires isort and black)
make format

# Run test suites
cd test/lang && python run_suite.py --suite per-commit
cd test/srt && python run_suite.py --suite per-commit

# Run individual tests with pytest
pytest test/lang/test_*.py
pytest test/srt/test_*.py

# Run kernel tests
cd sgl-kernel && pytest tests/
```

### Development Tools
```bash
# Benchmark performance
python python/sglang/bench_offline_throughput.py
python python/sglang/bench_serving.py

# Check environment
python python/sglang/check_env.py

# Launch server for testing
python python/sglang/launch_server.py --model-path meta-llama/Llama-2-7b-chat-hf --port 30000
```

## Architecture Overview

SGLang is a fast serving framework for large language models with a modular architecture:

### Core Components

1. **Frontend Language** (`python/sglang/lang/`)
   - Intuitive programming interface for LLM applications
   - Supports chained generation calls, control flow, multi-modal inputs
   - Key files: `compiler.py`, `interpreter.py`, `ir.py`, `tracer.py`

2. **SRT Backend** (`python/sglang/srt/`)
   - SGLang Runtime - efficient model serving engine
   - Features: RadixAttention (prefix caching), zero-overhead scheduler, continuous batching
   - Key directories: `layers/`, `managers/`, `mem_cache/`, `distributed/`

3. **Kernel Library** (`sgl-kernel/`)
   - High-performance CUDA kernels for optimized inference
   - Custom attention, MoE, quantization, and sampling kernels
   - Built with CMake and CUDA

4. **Router** (`sgl-router/`)
   - Rust-based load balancer with prefill-decode disaggregation
   - Multiple routing algorithms and cache-aware policies

### Key Features

- **RadixAttention**: Automatic prefix caching for efficient repeated text generation
- **Continuous Batching**: Dynamic batching with zero overhead
- **Structured Outputs**: Constrained generation with JSON schema support
- **Multi-Modal Support**: Vision and language models (LLaVA, etc.)
- **Quantization**: FP4/FP8/INT4/AWQ/GPTQ support
- **Parallelism**: Tensor/pipeline/expert/data parallelism
- **Speculative Decoding**: Faster inference with draft models

### Model Support

The framework supports extensive model families:
- **Generative Models**: Llama, Qwen, DeepSeek, Kimi, GPT, Gemma, Mistral
- **Embedding Models**: e5-mistral, gte, mcdse
- **Reward Models**: Skywork
- **Multi-Modal Models**: LLaVA series, Qwen-VL, InternVL

### Development Patterns

1. **Model Integration**: Add new models in `python/sglang/srt/configs/` following existing patterns
2. **Kernel Development**: Add CUDA kernels in `sgl-kernel/csrc/` with corresponding Python bindings
3. **Testing**: Use pytest framework with test suites organized in `test/lang/` and `test/srt/`
4. **Benchmarking**: Use provided benchmark scripts in `benchmark/` directory

### Configuration

- Model configurations: `python/sglang/srt/configs/model_config.py`
- Server arguments: `python/sglang/srt/server_args.py`
- Global settings: `python/sglang/global_config.py`

### Build System

The project uses a multi-package structure:
- Main package: `python/` (setuptools)
- Kernel library: `sgl-kernel/` (scikit-build-core with CMake)
- Router: `sgl-router/` (setuptools-rust with PyO3 bindings)

### Python Module Structure

The `python/sglang/` directory contains the following modules:

**Core Modules:**
- `lang/` - Frontend language interface with compiler, interpreter, and backend integrations
- `srt/` - SGLang Runtime backend with layers, managers, and model implementations
- `test/` - Test suite for various components including attention and ops

**Language Module (`lang/`):**
- `backend/` - Backend integrations (Anthropic, OpenAI, LiteLLM, VertexAI)
- `compiler.py`, `interpreter.py`, `ir.py`, `tracer.py` - Core language components
- `api.py`, `chat_template.py`, `choices.py` - API and interface utilities

**SRT Module (`srt/`) - Detailed Structure:**

**Core Infrastructure:**
- `configs/` - Model configurations (ChatGLM, DBRX, DeepSeekVL2, InternVL, etc.) and device settings
- `managers/` - Core scheduling, tokenization, session management, and cache control
- `mem_cache/` - Advanced memory management with radix trees, chunk caching, and storage backends
- `models/` - Extensive model implementations (50+ models including Llama, Qwen, DeepSeek, Gemma, etc.)
- `layers/` - Neural network components with attention, MoE, quantization, and linear layers

**Distributed Computing:**
- `distributed/` - Multi-GPU and multi-node distributed training/inference
  - `device_communicators/` - Hardware-specific communication (CUDA, NPU, HPU, XPU)
  - Communication operations and parallel state management
- `disaggregation/` - Prefill-decode disaggregation with various backends
  - `ascend/`, `mooncake/`, `nixl/` - Platform-specific transfer engines
  - `base/`, `common/`, `fake/` - Abstract and utility implementations

**Serving and APIs:**
- `entrypoints/` - HTTP server, OpenAI API compatibility, and engine interfaces
  - `openai/` - Complete OpenAI API implementation (chat, completions, embedding, etc.)
- `connector/` - External service connectors (Redis, S3) with serialization
- `sampling/` - Advanced sampling with penalty libraries and custom processors

**Advanced Features:**
- `constrained/` - Grammar-based constrained generation (Outlines, XGrammar, LLaMA Guidance)
- `function_call/` - Function calling capabilities with format detectors
- `speculative/` - Speculative decoding with EAGLE implementations
- `lora/` - LoRA fine-tuning support with memory management
- `multimodal/` - Multi-modal processing with various model processors

**Optimization Layers:**
- `layers/attention/` - Multiple attention backends (FlashAttention, FlashInfer, Triton, etc.)
- `layers/moe/` - Mixture of Experts implementations (Cutlass, Triton, DeepEP)
- `layers/quantization/` - Comprehensive quantization support (AWQ, GPTQ, FP8, INT8, etc.)

**Utilities and Support:**
- `model_executor/` - Model execution with CUDA/NPU graph runners
- `model_loader/` - Efficient model loading and weight management
- `debug_utils/` - Debugging and comparison utilities
- `metrics/` - Performance monitoring and timing
- `eplb/` - Expert placement and load balancing for MoE models

**Supporting Modules:**
- `eval/` - Evaluation scripts for different benchmarks
- `bench_*.py` - Benchmarking utilities
- `launch_server.py` - Server launch script
- `global_config.py` - Global configuration settings

## Server Architecture Details

### KV Cache Memory Management

SGLang implements a sophisticated three-tier KV cache memory management system:

#### **Memory Allocation Hierarchy**

1. **ReqToTokenPool** (`mem_cache/memory_pool.py`)
   - Maps requests to their token locations
   - Size: `(size, max_context_len)` tensor
   - Uses free list for allocation

2. **TokenToKVPoolAllocator** (`mem_cache/allocator.py`)
   - Manages indices to KV cache data
   - Multiple allocator types:
     - `TokenToKVPoolAllocator`: Token-level allocation
     - `PagedTokenToKVPoolAllocator`: Page-aligned allocation
     - `SWATokenToKVPoolAllocator`: Sliding Window Attention

3. **KVCache Storage**
   - Physical KV cache storage implementations:
     - `MHATokenToKVPool`: Multi-head attention
     - `MLATokenToKVPool`: Multi-head Latent Attention
     - `SWAKVPool`: Separate pools for full/SWA layers
     - `DoubleSparseTokenToKVPool`: Sparse attention models

#### **Request Lifecycle**

1. **Allocation** (`managers/schedule_batch.py`):
   - Request slot allocation from `ReqToTokenPool`
   - Token slot allocation from `TokenToKVPoolAllocator`
   - Prefix matching via `RadixCache` for reuse

2. **Execution**:
   - KV cache populated during model forward pass
   - Radix cache locks active requests (`lock_ref > 0`)

3. **Release/Reuse**:
   - Immediate free for non-cacheable requests
   - Storage in radix tree for future reuse
   - LRU eviction when memory pressure is high

#### **Advanced Features**

- **Radix Cache**: Prefix caching with radix tree structure
- **Multi-tier Storage**: Host-device cache hierarchy (HiCache)
- **Memory Optimization**: Page alignment, CPU offloading, custom pools
- **Leak Detection**: Automatic memory leak detection and reporting

### KV Cache Allocation and Freeing Lifecycle

#### **1. Request Arrival and Initial Processing**
```python
# Step 1: Request reception and prefix matching
req_pool_indices = self.alloc_req_slots(bs)  # From ReqToTokenPool
ret = self.tree_cache.match_prefix(req.origin_input_ids)  # Check radix cache
if ret.device_indices is not None:
    req.prefix_indices = ret.device_indices  # Reuse existing KV cache
```

#### **2. KV Cache Allocation**
```python
# Step 2: Allocate token slots
out_cache_loc = self.token_to_kv_pool_allocator.alloc(num_tokens)

# Step 3: Update ReqToTokenPool mapping
self.req_to_token_pool.write(
    (req_pool_indices[i], slice(prefix_lens[i], seq_lens[i])),
    out_cache_loc[pt : pt + extend_lens[i]],  # KV cache indices
)
```

#### **3. KV Cache Population**
```python
# During model forward pass
layer.set_kv_buffer(
    locs=forward_batch.out_cache_loc,  # Token locations from ReqToTokenPool
    cache_k=new_k,
    cache_v=new_v,
)
```

#### **4. Request Processing**
```python
# Incremental decoding - update mapping for new tokens
self.req_to_token_pool.write(
    (self.req_pool_indices, locs),
    self.out_cache_loc.to(torch.int32)
)
```

#### **5. KV Cache Freeing**
```python
# Request completion
token_indices = self.req_to_token_pool.req_to_token[
    req.req_pool_idx, : seq_lens_cpu[idx]
]
self.token_to_kv_pool_allocator.free(token_indices)  # Free KV slots
self.req_to_token_pool.free(req.req_pool_idx)  # Free request slot
```

#### **6. Optional Radix Cache Storage**
```python
# Store completed request for future reuse
self.tree_cache.insert(req.origin_input_ids, req)
req.finished = True  # Don't free KV cache
```

#### **7. Memory Eviction (When Needed)**
```python
# LRU-based eviction when memory pressure is high
if node.lock_ref == 0:  # Node not in use
    kv_indices = self.req_to_token_pool.req_to_token[
        node.req.req_pool_idx, : node.value_len
    ]
    self.token_to_kv_pool_allocator.free(kv_indices)
    self.req_to_token_pool.free(node.req.req_pool_idx)
```

### Key Data Flow

- **Allocation**: Request → ReqToTokenPool slot → TokenToKVPool slots → Update mapping
- **Freeing**: Complete → Get indices → Free KV slots → Free request slot

### Special Cases

- **Prefix Reuse**: No allocation for matched prefixes
- **Chunked Processing**: Incremental allocation for streaming
- **Eviction**: LRU-based eviction under memory pressure
- **Disaggregated Serving**: Separate prefill/decode handling

## LayerCommunicator Class

The `LayerCommunicator` class in SGLang manages communication patterns for tensor parallelism and data parallelism within transformer layers, optimizing data movement and computation overlap.

### Overview

**Location**: `python/sglang/srt/layers/communicator.py`

**Purpose**: Handles efficient communication between GPUs in distributed inference scenarios, managing tensor distribution patterns, all-reduce operations, and communication-computation overlap.

### Key Concepts

#### Scatter Modes
- `SCATTERED`: Each rank has its portion of data
- `TP_ATTN_FULL`: All ranks in tensor parallel group have full attention data
- `FULL`: All ranks have complete data

#### Communication Functions
The class uses three specialized communication function objects:
1. **Simple Communication**: Basic tensor redistribution
2. **All-Reduce with LayerNorm**: Fused operations for efficiency
3. **Summable Tensor Pair**: Handles (hidden_states, residual) pairs

### Class Methods

#### `__init__(layer_scatter_modes, input_layernorm, post_attention_layernorm, allow_reduce_scatter, is_last_layer)`
- Initializes communicator with layer-specific scatter modes
- Sets up communication function objects based on modes
- Configures reduce-scatter support and layer position

#### `prepare_attn(hidden_states, residual, forward_batch, qaunt_format)`
- Prepares tensors for attention computation
- Applies input layer norm with optional all-reduce fusion
- Handles MXFP4 quantization for ROCm devices
- Ensures proper tensor distribution across ranks

#### `prepare_mlp(hidden_states, residual, forward_batch)`
- Prepares tensors for MLP/MoE computation
- Handles post-attention layer norm
- Manages all-reduce operations fused with normalization
- Optimizes for sparse (MoE) vs dense layers

#### `postprocess_layer(hidden_states, residual, forward_batch)`
- Processes layer output after MLP/MoE computation
- Manages summation of hidden states and residual
- Handles final communication patterns for layer output
- Supports reduce-scatter optimization

#### `should_use_reduce_scatter(forward_batch)`
- Determines if reduce-scatter should replace all-reduce
- Checks: permission, function type, and padding mode
- Optimizes communication for padded batches

#### `should_fuse_mlp_allreduce_with_next_layer(forward_batch)`
- Decides whether to fuse MLP all-reduce with next layer's input norm
- Considers: layer position, TP size, FlashInfer availability, batch size
- Avoids fusion with EAGLE speculative decoding

### Usage Pattern

```python
# In transformer layer forward pass:
# 1. Prepare for attention
hidden_states, residual = layer_comm.prepare_attn(
    hidden_states, residual, forward_batch
)

# 2. Attention computation
attn_output = self_attn(hidden_states)

# 3. Prepare for MLP
mlp_input, mlp_residual = layer_comm.prepare_mlp(
    attn_output, residual, forward_batch
)

# 4. MLP computation
mlp_output = mlp(mlp_input)

# 5. Postprocess layer
hidden_states, residual = layer_comm.postprocess_layer(
    mlp_output, mlp_residual, forward_batch
)
```

### Performance Optimizations

1. **Communication Fusion**: Combines all-reduce with layer norm
2. **Dynamic Pattern Selection**: Chooses optimal communication based on layer type
3. **Hardware Awareness**: Leverages FlashInfer for GPU optimization
4. **Overlap Maximization**: Strategic yielding for computation-communication overlap
5. **Memory Efficiency**: Reduce-scatter when beneficial

### Integration

The LayerCommunicator is essential for:
- Tensor parallel inference efficiency
- MoE model performance
- Multi-GPU communication optimization
- Hardware-specific acceleration

## Operations Strategy for MoE Models

The `operations_strategy.py` file defines execution strategies for MoE (Mixture of Experts) models, specifically optimized for Tensor Parallelism with Blocking Overlap (TBO) to maximize GPU utilization.

### Overview

**Location**: `python/sglang/srt/operations_strategy.py`

**Purpose**: Defines optimized operation sequences for MoE model layers, strategically placing yield points to overlap computation with communication and balance GPU resources.

### Key Components

#### OperationsStrategy Class
A dataclass containing:
- `operations`: List of operations to execute in sequence
- `deep_gemm_num_sms`: Number of SMs allocated for deep GEMM operations
- `tbo_delta_stages`: Number of delta stages for TBO synchronization

**Key Methods**:
- `concat()`: Combines multiple strategies into one
- `init_new_tbo()`: Creates TBO strategies for specific model types

### Model-Specific Strategies

#### DeepSeek MoE Strategy
1. **Prefill Mode** (`_compute_moe_deepseek_blog_prefill`)
   - Reserves SMs for DeepEP communication
   - Sequence: attention prep → attention → MoE gate → dispatch → yield → experts → yield → shared experts → output
   - 2 strategic yield points for optimal overlap

2. **Decode Mode** (`_compute_moe_deepseek_blog_decode`)
   - More aggressive yielding (4 points)
   - 2 delta stages for better TBO synchronization
   - Distributed operations with fine-grained overlap

#### Qwen3 MoE Strategy
Similar to DeepSeek but with optimizations:
- Fewer yield points in decode mode
- No shared experts architecture
- Slightly different operation ordering

### Key Features

1. **Tensor Parallelism with Blocking Overlap (TBO)**
   - Strategic `YieldOperation()` placement for computation-communication overlap
   - Delta stages control synchronization granularity
   - Maximizes GPU utilization during communication

2. **Dynamic Resource Management**
   - Allocates SMs between computation (deep_gemm) and communication (DeepEP)
   - Calculated based on device properties: `total_sms - deepep_sms`

3. **Mode-Specific Optimization**
   - Different strategies for prefill (EXTEND) vs decode (DECODE) phases
   - Prefill: Focus on throughput with fewer yields
   - Decode: Focus on latency with more fine-grained yielding

4. **Hardware-Aware Design**
   - Considers GPU SM count for resource allocation
   - Optimized for NVIDIA GPUs with CUDA
   - Balances load across available resources

### Usage Example

```python
# Create strategy for a layer
strategy = OperationsStrategy.init_new_tbo(
    layers=model_layers,
    forward_mode=ForwardMode.EXTEND  # or DECODE
)

# Execute operations
for op in strategy.operations:
    if isinstance(op, YieldOperation):
        yield  # Allow other operations to run
    else:
        result = op()
```

### Operation Flow (DeepSeek Decode)
1. Communication preparation for attention
2. Attention preparation → yield
3. Attention core computation
4. MLP communication prep
5. MoE gate computation → expert selection
6. Token dispatch A → shared experts → yield
7. Token dispatch B → expert computation → combine → yield
8. Final combine → yield → output → postprocess

### Performance Benefits

1. **Improved Utilization**: Overlaps communication with computation
2. **Better Balance**: Dynamic SM allocation based on workload
3. **Reduced Latency**: Strategic yielding minimizes stalls
4. **Scalability**: Designed for large-scale tensor parallel deployments

### Integration Points

- Used by MoE model layers during forward pass
- Integrates with DeepEP for expert communication
- Works with LayerCommunicator for tensor parallelism
- Supports FlashInfer optimizations where available

## SGLang Server Command-Line Arguments

SGLang provides over 180 command-line arguments to configure the server behavior. The arguments are defined in `python/sglang/srt/server_args.py` and can be viewed using `python -m sglang.launch_server --help`.

### Model and Tokenizer Arguments

- **`--model-path, --model`** (required): Path to model weights (local folder or Hugging Face repo ID)
- **`--tokenizer-path`**: Path to the tokenizer
- **`--tokenizer-worker-num`**: Number of tokenizer manager workers
- **`--tokenizer-mode`**: Tokenizer mode (auto/slow)
- **`--skip-tokenizer-init`**: Skip tokenizer initialization
- **`--load-format`**: Model weight format (auto/pt/safetensors/npcache/dummy/gguf/bitsandbytes/layered)
- **`--trust-remote-code`**: Allow custom model definitions from Hub
- **`--context-length`**: Model's maximum context length
- **`--revision`**: Specific model version (branch/tag/commit)

### Server Configuration

- **`--host`**: HTTP server host
- **`--port`**: HTTP server port
- **`--device`**: Device to use (cuda/xpu/hpu/npu/cpu)
- **`--tensor-parallel-size, --tp-size`**: Tensor parallelism size
- **`--pipeline-parallel-size, --pp-size`**: Pipeline parallelism size
- **`--data-parallel-size, --dp-size`**: Data parallelism size
- **`--dist-init-addr`**: Distributed backend initialization address
- **`--nnodes`**: Number of nodes in multi-node setup
- **`--node-rank`**: Node rank in multi-node setup

### Memory and Performance

- **`--mem-fraction-static`**: Memory fraction for static allocation
- **`--max-running-requests`**: Maximum concurrent requests
- **`--max-queued-requests`**: Maximum queued requests
- **`--max-total-tokens`**: Maximum tokens in memory pool
- **`--chunked-prefill-size`**: Maximum tokens per prefill chunk (-1 = disable)
- **`--max-prefill-tokens`**: Maximum tokens in prefill batch
- **`--schedule-policy`**: Request scheduling policy (lpm/random/fcfs/dfs-weight/lof)
- **`--schedule-conservativeness`**: Scheduling conservativeness (higher = more conservative)
- **`--page-size`**: Tokens per page in paged attention

### Attention and KV Cache

- **`--attention-backend`**: Attention kernel backend
- **`--prefill-attention-backend`**: Prefill-specific attention backend
- **`--decode-attention-backend`**: Decode-specific attention backend
- **`--kv-cache-dtype`**: KV cache data type (auto/fp8_e5m2/fp8_e4m3)
- **`--disable-radix-cache`**: Disable prefix caching
- **`--hybrid-kvcache-ratio`**: Mix ratio between uniform and hybrid KV buffers
- **`--enable-hierarchical-cache`**: Enable hierarchical caching
- **`--hicache-ratio`**: Host KV cache size ratio to device
- **`--hicache-storage-backend`**: Storage backend (file/mooncake/hf3fs/nixl)

### Quantization

- **`--dtype`**: Data type for weights/activations (auto/half/float16/bfloat16/float32)
- **`--quantization`**: Quantization method
- **`--quantization-param-path`**: Path to KV cache scaling factors
- **`--torchao-config`**: TorchAO optimization config (int8dq/int8wo/int4wo/fp8wo)

### Kernel Backends

- **`--sampling-backend`**: Sampling kernel backend (flashinfer/pytorch)
- **`--grammar-backend`**: Grammar-guided decoding backend
- **`--mm-attention-backend`**: Multimodal attention backend (sdpa/fa3/triton_attn)
- **`--disable-cuda-graph`**: Disable CUDA graph optimization
- **`--enable-torch-compile`**: Enable torch.compile optimization

### LoRA Support

- **`--enable-lora`**: Enable LoRA support
- **`--lora-paths`**: List of LoRA adapters to load
- **`--max-lora-rank`**: Maximum LoRA adapter rank
- **`--lora-target-modules`**: Target modules for LoRA
- **`--max-loras-per-batch`**: Maximum LoRAs per batch
- **`--max-loaded-loras`**: Maximum loaded LoRAs in CPU

### Speculative Decoding

- **`--speculative-algorithm`**: Algorithm (EAGLE/EAGLE3/NEXTN/STANDALONE)
- **`--speculative-draft-model-path`**: Draft model path
- **`--speculative-num-steps`**: Number of draft steps
- **`--speculative-eagle-topk`**: Top-k for EAGLE sampling
- **`--speculative-accept-threshold-single`**: Single token acceptance threshold
- **`--speculative-accept-threshold-acc`**: Accumulative acceptance threshold

### MoE and Expert Parallelism

- **`--expert-parallel-size, --ep-size, --ep`**: Expert parallelism size
- **`--moe-a2a-backend`**: MoE A2A backend (none/deepep)
- **`--moe-runner-backend`**: MoE runner backend
- **`--enable-flashinfer-allreduce-fusion`**: Enable all-reduce fusion
- **`--deepep-mode`**: DeepEP mode (normal/low_latency/auto)
- **`--enable-eplb`**: Enable EPLB load balancing
- **`--ep-num-redundant-experts`**: Number of redundant experts

### Data Parallel Attention

- **`--enable-dp-attention`**: Enable data parallel for attention
- **`--enable-dp-lm-head`**: Enable vocabulary parallel across DP groups
- **`--load-balance-method`**: Load balancing strategy (round_robin/shortest_queue/minimum_tokens)

### Multimodal and Tool Use

- **`--enable-multimodal`**: Enable multimodal functionality
- **`--tool-call-parser`**: Tool call parser (none/llama3_tool_call/qwen_tool_call/etc)
- **`--tool-server`**: Tool server URLs
- **`--reasoning-parser`**: Reasoning model parser (deepseek_r1/none)
- **`--mm-attention-backend`**: Multimodal attention backend

### Disaggregation

- **`--disaggregation-mode`**: PD disaggregation mode (prefill/decode)
- **`--disaggregation-transfer-backend`**: Transfer backend
- **`--disaggregation-bootstrap-port`**: Bootstrap server port
- **`--disaggregation-decode-tp`**: Decode TP size (prefill server only)
- **`--disaggregation-decode-dp`**: Decode DP size (prefill server only)

### Logging and Monitoring

- **`--log-level`**: Logging level
- **`--log-requests`**: Log request metadata and I/O
- **`--log-requests-level`**: Request logging verbosity (0-3)
- **`--enable-metrics`**: Enable Prometheus metrics
- **`--show-time-cost`**: Show custom timing marks
- **`--crash-dump-folder`**: Folder for crash dumps
- **`--collect-tokens-histogram`**: Collect token statistics

### Optimization Features

- **`--enable-mixed-chunk`**: Mix prefill and decode in batch
- **`--enable-two-batch-overlap`**: Enable two micro-batch overlap
- **`--num-continuous-decode-steps`**: Continuous decode steps (default: 1)
- **`--disable-overlap-schedule`**: Disable CPU-GPU overlap
- **`--cuda-graph-max-bs`**: Maximum CUDA graph batch size
- **`--enable-nan-detection`**: Enable NaN detection for debugging
- **`--enable-memory-saver`**: Enable memory saving features

### API Configuration

- **`--api-key`**: Server API key
- **`--served-model-name`**: Model name for API responses
- **`--chat-template`**: Chat template path or name
- **`--completion-template`**: Completion template
- **`--file-storage-path`**: File storage path
- **`--enable-cache-report`**: Report cached tokens in usage stats

### Debug and Development

- **`--skip-server-warmup`**: Skip server warmup
- **`--warmups`**: Custom warmup functions
- **`--watchdog-timeout`**: Watchdog timeout in seconds
- **`--random-seed`**: Random seed for reproducibility
- **`--delete-ckpt-after-loading`**: Delete checkpoint after loading
- **`--allow-auto-truncate`**: Auto-truncate long inputs
- **`--debug-tensor-dump-output-folder`**: Tensor dump output folder

### Usage Examples

```bash
# Basic server launch
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --port 30000

# Multi-GPU inference
python -m sglang.launch_server --model-path meta-llama/Llama-2-70b-chat-hf --tp-size 4

# Enable LoRA support
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --enable-lora --lora-paths adapter1=path/to/adapter1

# Quantized model
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --quantization fp8

# Speculative decoding
python -m sglang.launch_server --model-path meta-llama/Llama-2-7b-chat-hf --speculative-algorithm EAGLE --speculative-draft-model-path draft-model

# View all options
python -m sglang.launch_server --help
```

For the complete and most up-to-date list of arguments, always refer to the help output or the source code in `server_args.py`.