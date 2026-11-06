# Shared CUDA + Python environment. Child images build on top of this one.
# The -devel image is required: torchmcubes is compiled with nvcc at install time.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# git: torchmcubes is installed from a git URL. libgl1/libglib2.0-0: runtime deps of image libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /uvx /usr/local/bin/

# No GPU is visible during `docker build`, so torchmcubes cannot detect the architecture;
# list the targets explicitly (override with --build-arg for other GPUs).
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    CUDA_HOME=/usr/local/cuda \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    PATH="/opt/venv/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --extra cuda --frozen --no-install-project
