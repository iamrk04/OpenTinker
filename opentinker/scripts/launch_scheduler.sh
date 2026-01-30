#!/bin/bash
# =============================================================================
# OpenTinker Job Scheduler Launch Script
# =============================================================================
# Auto-detects CUDA, GPUs, and other settings
# Usage: ./launch_scheduler.sh [options]
# =============================================================================

set -e

# =============================================================================
# Auto-detect CUDA_HOME
# =============================================================================
auto_detect_cuda() {
    # Priority order for CUDA detection:
    # 1. Already set CUDA_HOME
    # 2. nvcc location
    # 3. Common paths

    if [ -n "$CUDA_HOME" ] && [ -d "$CUDA_HOME" ]; then
        echo "Using existing CUDA_HOME: $CUDA_HOME"
        return 0
    fi

    # Try to find nvcc
    if command -v nvcc &> /dev/null; then
        NVCC_PATH=$(which nvcc)
        CUDA_HOME=$(dirname $(dirname $NVCC_PATH))
        echo "Auto-detected CUDA from nvcc: $CUDA_HOME"
        return 0
    fi

    # Check common paths
    COMMON_PATHS=(
        "/usr/local/cuda"
        "/usr/local/cuda-12"
        "/usr/local/cuda-12.8"
        "/usr/local/cuda-12.4"
        "/usr/local/cuda-12.2"
        "/usr/local/cuda-11.8"
        "$HOME/local/cuda"
        "$HOME/local/cuda-12.8"
        "/opt/cuda"
    )

    for path in "${COMMON_PATHS[@]}"; do
        if [ -d "$path" ] && [ -f "$path/bin/nvcc" ]; then
            CUDA_HOME="$path"
            echo "Auto-detected CUDA at: $CUDA_HOME"
            return 0
        fi
    done

    echo "WARNING: Could not auto-detect CUDA_HOME. Please set it manually."
    echo "         export CUDA_HOME=/path/to/cuda"
    return 1
}

# =============================================================================
# Auto-detect available GPUs
# =============================================================================
auto_detect_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        # Get number of GPUs
        NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
        if [ "$NUM_GPUS" -gt 0 ]; then
            # Build array [0,1,2,...,N-1]
            GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))
            echo "[$GPU_LIST]"
            return 0
        fi
    fi
    # Default fallback
    echo "[0,1,2,3]"
}

# =============================================================================
# Auto-detect GPU architecture for TORCH_CUDA_ARCH_LIST
# =============================================================================
auto_detect_gpu_arch() {
    if command -v nvidia-smi &> /dev/null; then
        # Get compute capability from first GPU
        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
        if [ -n "$COMPUTE_CAP" ]; then
            echo "$COMPUTE_CAP"
            return 0
        fi
    fi
    # Default fallback (Hopper H100)
    echo "9.0"
}

# =============================================================================
# Setup environment
# =============================================================================
echo "========================================"
echo "OpenTinker Job Scheduler"
echo "========================================"
echo ""
echo "[1/4] Detecting CUDA environment..."
auto_detect_cuda
export CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export NVCC_EXECUTABLE=$CUDA_HOME/bin/nvcc
echo "  CUDA_HOME: $CUDA_HOME"
echo "  NVCC: $NVCC_EXECUTABLE"

echo ""
echo "[2/4] Detecting GPU configuration..."
GPU_ARCH=$(auto_detect_gpu_arch)
export TORCH_CUDA_ARCH_LIST="$GPU_ARCH"
echo "  GPU Architecture: $GPU_ARCH"

# Auto-detect available GPUs if not specified
DEFAULT_GPUS=$(auto_detect_gpus)
echo "  Available GPUs: $DEFAULT_GPUS"

echo ""
echo "[3/4] Setting up environment variables..."
# Set trace directory relative to script location
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export ROLLOUT_TRACE_DIR="${ROLLOUT_TRACE_DIR:-$REPO_ROOT/traces}"
export FLASHINFER_HOMOGENEOUS_MS=1
mkdir -p "$ROLLOUT_TRACE_DIR" 2>/dev/null || true
echo "  Trace directory: $ROLLOUT_TRACE_DIR"

# Default configuration
AVAILABLE_GPUS="${DEFAULT_GPUS}"
PORT_RANGE="null"  # Set to null for auto-detection
NUM_PORTS=200
SCHEDULER_PORT=8780

echo ""
echo "[4/4] Parsing command line arguments..."

# Parse command line arguments (optional)
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            AVAILABLE_GPUS="$2"
            shift 2
            ;;
        --ports)
            PORT_RANGE="$2"
            shift 2
            ;;
        --num-ports)
            NUM_PORTS="$2"
            shift 2
            ;;
        --scheduler-port)
            SCHEDULER_PORT="$2"
            shift 2
            ;;
        --auto-ports)
            PORT_RANGE="null"
            shift 1
            ;;
        --help|-h)
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpus '[0,1,2,3]'    Specify GPUs (default: auto-detect)"
            echo "  --ports '[start,end]' Port range (default: auto-detect)"
            echo "  --auto-ports          Use auto port detection"
            echo "  --num-ports N         Number of ports for auto mode (default: 200)"
            echo "  --scheduler-port N    Scheduler API port (default: 8780)"
            echo "  --help, -h            Show this help"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo ""
echo "========================================"
echo "Configuration Summary"
echo "========================================"
echo "  CUDA_HOME:        $CUDA_HOME"
echo "  GPU Architecture: $TORCH_CUDA_ARCH_LIST"
echo "  Available GPUs:   $AVAILABLE_GPUS"
if [ "$PORT_RANGE" = "null" ]; then
    echo "  Port Mode:        Auto-detect ($NUM_PORTS ports)"
else
    echo "  Port Range:       $PORT_RANGE"
fi
echo "  Scheduler Port:   $SCHEDULER_PORT"
echo "  Trace Directory:  $ROLLOUT_TRACE_DIR"
echo "========================================"
echo ""
echo "Starting scheduler..."
echo ""

# Change to repo root to run python module
cd "$REPO_ROOT"

# Launch scheduler
if [ "$PORT_RANGE" = "null" ]; then
    python -m opentinker.scheduler.launch_scheduler_kill \
        available_gpus=$AVAILABLE_GPUS \
        port_range=null \
        num_ports=$NUM_PORTS \
        scheduler_port=$SCHEDULER_PORT
else
    python -m opentinker.scheduler.launch_scheduler_kill \
        available_gpus=$AVAILABLE_GPUS \
        port_range=$PORT_RANGE \
        scheduler_port=$SCHEDULER_PORT
fi
