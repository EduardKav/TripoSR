# TripoSR

<a href="https://www.tripo3d.ai/research/triposr"><img alt="Technical blog post on tripo3d.ai" src="https://img.shields.io/badge/Technical_Blog-gray?logo=data:image/svg%2bxml;base64,PHN2ZyB3aWR0aD0iNjUiIGhlaWdodD0iNjUiIHZpZXdCb3g9IjAgMCA2NSA2NSIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTkuNDk5MSA5LjYzNDc3TDE2LjQzNzQgMjEuNDU1NkMxNi40MzkzIDIxLjQ1ODkgMTYuNDQxMiAyMS40NjIyIDE2LjQ0MzEgMjEuNDY1NUwzMC4yNjU4IDQ1LjA1NDhDMzEuNTMyNyA0Ny4yMTY3IDM0LjcwNDUgNDcuMjE2NyAzNS45NzE0IDQ1LjA1NDhMNDkuMzg2MiAyMi4xNjE2SDU5LjQ2MThMNDEuMjY2IDUzLjE2MkMzNy42NDQ5IDU5LjMzMTMgMjguNTkyMyA1OS4zMzEzIDI0Ljk3MTIgNTMuMTYyTDYuNjM5NjcgMjEuOTMwMkM0LjAyNCAxNy40NzM5IDUuNjU5NTYgMTIuMjEyNyA5LjQ5OTEgOS42MzQ3N1oiIGZpbGw9IndoaXRlIi8+CjxwYXRoIGQ9Ik0yMC4xMTIxIDE2LjYwODdIMzQuNjkyNkwyOC42MjIgMjcuMDQ0MkMyOC4yMDMzIDI3Ljc2NCAyOC4yMDgzIDI4LjY0OTIgMjguNjM1MSAyOS4zNjQ0TDMxLjA1MjcgMzMuNDE1MUMzMS45NjU0IDM0Ljk0NDUgMzQuMjE2MyAzNC45MzY1IDM1LjExNzggMzMuNDAwNkw0NC45NzM5IDE2LjYwODdINDYuOTQyTDQ2Ljk0NTUgMTYuNjA4N0g2MC44NDQ2QzYwLjQ4MzIgMTIuMDU4NyA1Ni42NzMxIDguMDQ4ODMgNTEuNDUwOSA4LjA0ODgzTDE1LjA4NzkgOC4wNDg4M0wyMC4xMTIxIDE2LjYwODdaIiBmaWxsPSIjRjhDRjAwIi8+Cjwvc3ZnPgo="></a> <a href="https://huggingface.co/spaces/stabilityai/TripoSR"><img alt="Try the TripoSR demo on Hugging Face Spaces" src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Demo-orange"></a> <a href="https://arxiv.org/abs/2403.02151"><img alt="Read the paper on arXiv (2403.02151)" src="https://img.shields.io/badge/Arxiv-2403.02151-B31B1B.svg"></a> <a href="https://discord.gg/mvS9mCfMnQ"><img alt="Join the Tripo AI Discord" src="https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white"></a>

<div align="center">
  <img src="figures/teaser800.gif" alt="Animated examples of TripoSR turning single photos into rotating 3D models">
</div>

This is the official codebase for **TripoSR**, a state-of-the-art open-source model for **fast** feedforward 3D reconstruction from a single image, collaboratively developed by [Tripo AI](https://www.tripo3d.ai/) and [Stability AI](https://stability.ai/).
<br><br>
Leveraging the principles of the [Large Reconstruction Model (LRM)](https://yiconghong.me/LRM/), TripoSR brings to the table key advancements that significantly boost both the speed and quality of 3D reconstruction. Our model is distinguished by its ability to rapidly process inputs, generating high-quality 3D models in less than 0.5 seconds on an NVIDIA A100 GPU. TripoSR has exhibited superior performance in both qualitative and quantitative evaluations, outperforming other open-source alternatives across multiple public datasets. The figures below illustrate visual comparisons and metrics showcasing TripoSR's performance relative to other leading models. Details about the model architecture, training process, and comparisons can be found in this [technical report](https://arxiv.org/abs/2403.02151).

<p align="center">
    <img width="800" src="figures/visual_comparisons.jpg" alt="Visual comparison of 3D meshes reconstructed from single images by ZeroShape, TGS, One-2-3-45, OpenLRM and TripoSR"/>
</p>

<p align="center">
    <img width="450" src="figures/scatter-comparison.png" alt="Scatter plot comparing TripoSR's reconstruction quality and inference speed against other open-source methods"/>
</p>

The model is released under the MIT license, which includes the source code, pretrained models, and an interactive online demo. Our goal is to empower researchers, developers, and creatives to push the boundaries of what's possible in 3D generative AI and 3D content creation.

## Getting Started

### Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`brew install uv`, or see the [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/)). uv installs Python 3.12 automatically if needed (Python >= 3.11 is required).

PyTorch is provided through two mutually exclusive extras. Pick the one that matches your machine, so CUDA libraries are only downloaded where you have an NVIDIA GPU:

| Setup                 | Command                | PyTorch                      |
| --------------------- | ---------------------- | ---------------------------- |
| macOS (Apple Silicon) | `uv sync --extra cpu`  | CPU/MPS build                |
| Linux, no NVIDIA GPU  | `uv sync --extra cpu`  | CPU build (no CUDA download) |
| Linux with NVIDIA GPU | `uv sync --extra cuda` | `2.6.0+cu124`                |

Notes:

- Always pass one of the extras; a plain `uv sync` installs no PyTorch.
- **CUDA:** building `torchmcubes` with GPU support requires the CUDA toolkit (`nvcc`, CUDA 12.x to match the PyTorch build) to be installed locally. Without it, `torchmcubes` is built CPU-only (see Troubleshooting).
- Supported platforms are macOS arm64 and Linux. Windows and Intel Macs are not covered by the lockfile.
- Run any command inside the environment with `uv run`, e.g. `uv run triposr ...`.

### Manual Inference

```sh
uv run triposr examples/chair.png --output-dir output/
```

This will save the reconstructed 3D model to `output/`. You can also specify more than one image path separated by spaces. The default options takes about **6GB VRAM** for a single image input.

If you would like to output a texture instead of vertex colors, use the `--bake-texture` option. You may also use `--texture-resolution` to specify the resolution in pixels of the output texture.

For detailed usage of this command, use `uv run triposr --help`.

### Local Gradio App

```sh
uv run python gradio_app.py
```

## Docker

Docker images are provided for **Linux with an NVIDIA GPU** only. The host needs an up-to-date NVIDIA driver (CUDA 12.4 compatible) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). No local CUDA toolkit is required.

### Available images

| Image            | Dockerfile                 | Description                                                                                                                  |
| ---------------- | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `triposr-base`   | `docker/base.Dockerfile`   | Shared base: CUDA 12.4 (devel), uv, and the project dependencies (`uv sync --extra cuda`) with `torchmcubes` built for CUDA. |
| `triposr-gradio` | `docker/gradio.Dockerfile` | Gradio web app built on `triposr-base`. Serves on port 7860.                                                                 |

`triposr-gradio` builds on `triposr-base`, so the base must be built first. Docker Compose does this automatically.

### Build and run with Docker Compose

```sh
docker compose up --build gradio
```

Then open http://localhost:7860. To only build the images:

```sh
docker compose build gradio   # also builds triposr-base if needed
```

Model weights are downloaded from the Hugging Face hub on first start and stored in the `hf_cache` volume, so later starts reuse them.

### Build without Compose

```sh
docker build -f docker/base.Dockerfile -t triposr-base .
docker build -f docker/gradio.Dockerfile -t triposr-gradio .
docker run --rm --gpus all -p 7860:7860 -v hf_cache:/cache/huggingface triposr-gradio
```

### Build options

`torchmcubes` is compiled during the build, when no GPU is visible, so the target GPU architectures are set explicitly (default: `7.5;8.0;8.6;8.9;9.0`). If your GPU is not covered, override it:

```sh
docker build -f docker/base.Dockerfile --build-arg TORCH_CUDA_ARCH_LIST="8.6" -t triposr-base .
```

## Troubleshooting

> AttributeError: module 'torchmcubes_module' has no attribute 'mcubes_cuda'

or

> torchmcubes was not compiled with CUDA support, use CPU version instead.

This is because `torchmcubes` is compiled without CUDA support. Please make sure that

- You installed with `--extra cuda` (not `--extra cpu`) on a machine with an NVIDIA GPU.
- The locally-installed CUDA toolkit major version (`nvcc --version`) matches the PyTorch-shipped CUDA major version (12.x for `--extra cuda`).

Then rebuild `torchmcubes`:

```sh
uv sync --extra cuda --reinstall-package torchmcubes
```

## Citation

```BibTeX
@article{TripoSR2024,
  title={TripoSR: Fast 3D Object Reconstruction from a Single Image},
  author={Tochilkin, Dmitry and Pankratz, David and Liu, Zexiang and Huang, Zixuan and and Letts, Adam and Li, Yangguang and Liang, Ding and Laforte, Christian and Jampani, Varun and Cao, Yan-Pei},
  journal={arXiv preprint arXiv:2403.02151},
  year={2024}
}
```
