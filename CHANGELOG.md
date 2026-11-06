# Changelog

All notable changes to this fork are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Changes before this fork's first entry belong to upstream
[VAST-AI-Research/TripoSR](https://github.com/VAST-AI-Research/TripoSR) and are not tracked here.

## [1.0.0] - 2026-11-06

### Added

- `triposr` command-line tool (`tsr/cli.py`, built with typer), with new `--render-views` and `--render-fps` options.
- uv-based dependency management (`pyproject.toml`, `uv.lock`, `.python-version`) with mutually exclusive `cpu` and `cuda` PyTorch extras.
- Docker images and `compose.yaml` for the CLI and Gradio app on Linux (CUDA).
- Pre-commit hooks: prettier, black and isort.
- "About this fork" section in the README.

### Changed

- Heavy imports are loaded lazily to speed up CLI startup.
- Timing in the CLI uses a context manager instead of a `Timer` class.
- README installation instructions now describe the uv workflow.
- Added the `fastapi` dependency to fix the Gradio app.

### Fixed

- `--device` override was not respected.
- Crash on export when using `--no-remove-bg`.

### Removed

- `run.py`, replaced by the `triposr` command.
- `requirements.txt`, replaced by `pyproject.toml`.

[1.0.0]: https://github.com/EduardKav/TripoSR/releases/tag/v1.0.0
