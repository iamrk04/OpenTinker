# =============================================================================
# OpenTinker Docker Image
# =============================================================================
# 
# BUILD (maintainers only):
#   docker build -t opentinker-env:latest .
#
# PUSH TO ACR (maintainers only):
#   az acr login --name rltrainingacr
#   docker tag opentinker-env:latest rltrainingacr.azurecr.io/opentinker-env:latest
#   docker push rltrainingacr.azurecr.io/opentinker-env:latest
#
# =============================================================================
# FOR USERS: One-time setup on Azure Compute Instance
# =============================================================================
#
# STEP 1 - Pull image (one-time):
#   docker pull rltrainingacr.azurecr.io/opentinker-env:latest
#
# STEP 2 - Create persistent container (one-time):
#   docker run -d --gpus all --restart=unless-stopped --entrypoint /bin/bash --net=host --shm-size=40g --cap-add=SYS_ADMIN  -v /home/azureuser/workspace:/workspace/dev --name tinker rltrainingacr.azurecr.io/opentinker-env:latest -c "tail -f /dev/null"
#
# DAILY USAGE (after CI restart, container auto-starts):
#   docker exec -it tinker bash
#
# =============================================================================

FROM verlai/verl@sha256:3ce56ff018516b28ab9c4f4fc09d3aa67589074495ace75e2674b720aa4d0e5d

LABEL maintainer="RL Team"
LABEL description="OpenTinker - Multi-turn RL Training for LLMs"
LABEL version="1"

# Set working directory
WORKDIR /workspace

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Clone OpenTinker repository with submodules
RUN git clone --recurse-submodules -b rk/fix https://github.com/iamrk04/OpenTinker.git /workspace/OpenTinker

WORKDIR /workspace/OpenTinker

# Install OpenTinker
RUN pip install -e . --no-cache-dir

# Install verl submodule
RUN cd verl && pip install -e . --no-cache-dir && cd ..

# Install specific package versions
RUN pip install transformers==4.57.6 job_scheduler --no-cache-dir

# Install flash-attention (pre-built wheel for CUDA 12 + PyTorch 2.8)
RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl --no-cache-dir

# Verify installation
RUN python -c "import opentinker; import verl; import flash_attn; import transformers; print('✅ All packages installed successfully!')"

# Create outputs directory
RUN mkdir -p /workspace/OpenTinker/outputs

# Default command - start interactive bash
CMD ["/bin/bash"]
