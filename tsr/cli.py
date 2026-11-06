import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
import xatlas
from PIL import Image

# torch, rembg and the tsr package are imported inside the functions that use
# them, so that `--help` and argument validation errors don't pay for importing them.


def _synchronize(device: str | None) -> None:
    """Wait for queued work on the device so timings reflect finished work."""
    kind = (device or "").split(":")[0]
    if kind == "cuda":
        import torch

        torch.cuda.synchronize()
    elif kind == "mps":
        import torch

        torch.mps.synchronize()


@contextmanager
def timed(name: str, device: str | None = None):
    """Log the wall-clock duration of the enclosed block, in milliseconds.

    Pass the torch device the block runs on so asynchronous GPU work is included.
    """
    logging.info(f"{name} ...")
    _synchronize(device)
    start = time.perf_counter()
    yield
    _synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    logging.info(f"{name} finished in {elapsed_ms:.2f}ms.")


class ModelFormat(str, Enum):
    obj = "obj"
    glb = "glb"


def resolve_device(requested: str) -> str:
    """Return a usable torch device, falling back when the requested one is missing.

    An unavailable CUDA device falls back to MPS if present, otherwise CPU. An
    unavailable MPS device falls back to CPU. Any other device is used as given.
    """
    import torch

    kind = requested.split(":")[0]
    if kind == "cuda" and torch.cuda.is_available():
        return requested
    if kind in ("cuda", "mps"):
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        if device != requested:
            logging.warning(f"Device '{requested}' is not available, using '{device}'.")
        return device
    return requested


def load_model(pretrained_model_name_or_path: str, chunk_size: int, device: str):
    from tsr.system import TSR

    with timed("Initializing model", device):
        model = TSR.from_pretrained(
            pretrained_model_name_or_path,
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.renderer.set_chunk_size(chunk_size)
        model.to(device)
    return model


def preprocess_images(
    image_paths: list[Path],
    output_dir: Path,
    no_remove_bg: bool,
    foreground_ratio: float,
) -> list:
    import rembg

    from tsr.utils import remove_background, resize_foreground

    images = []
    with timed("Processing images"):
        rembg_session = None if no_remove_bg else rembg.new_session()

        for i, image_path in enumerate(image_paths):
            os.makedirs(os.path.join(output_dir, str(i)), exist_ok=True)
            if no_remove_bg:
                input_image = Image.open(image_path).convert("RGB")
                img = np.array(input_image)
            else:
                img = remove_background(Image.open(image_path), rembg_session)
                img = resize_foreground(img, foreground_ratio)
                img = np.array(img).astype(np.float32) / 255.0
                img = img[:, :, :3] * img[:, :, 3:4] + (1 - img[:, :, 3:4]) * 0.5
                img = Image.fromarray((img * 255.0).astype(np.uint8))
                input_image = img
            # Save what the model actually sees, whichever preprocessing ran.
            input_image.save(os.path.join(output_dir, str(i), "input.png"))
            images.append(img)
    return images


def render_views(model, scene_codes, out_dir: str, n_views: int, fps: int) -> None:
    from tsr.utils import save_video

    with timed("Rendering", str(scene_codes.device)):
        render_images = model.render(scene_codes, n_views=n_views, return_type="pil")
        for ri, render_image in enumerate(render_images[0]):
            render_image.save(os.path.join(out_dir, f"render_{ri:03d}.png"))
        save_video(render_images[0], os.path.join(out_dir, "render.mp4"), fps=fps)


def export_baked_mesh(
    model, scene_codes, mesh, out_dir: str, out_mesh_path: str, texture_resolution: int
) -> None:
    from tsr.bake_texture import bake_texture

    out_texture_path = os.path.join(out_dir, "texture.png")

    with timed("Baking texture", str(scene_codes.device)):
        bake_output = bake_texture(mesh, model, scene_codes[0], texture_resolution)

    with timed("Exporting mesh and texture"):
        xatlas.export(
            out_mesh_path,
            mesh.vertices[bake_output["vmapping"]],
            bake_output["indices"],
            bake_output["uvs"],
            mesh.vertex_normals[bake_output["vmapping"]],
        )
        Image.fromarray((bake_output["colors"] * 255.0).astype(np.uint8)).transpose(
            Image.FLIP_TOP_BOTTOM
        ).save(out_texture_path)


def export_vertex_color_mesh(mesh, out_mesh_path: str) -> None:
    with timed("Exporting mesh"):
        mesh.export(out_mesh_path)


@dataclass(frozen=True)
class ProcessingOptions:
    """Per-image options controlling inference, rendering and export."""

    render: bool
    render_views: int
    render_fps: int
    bake_texture: bool
    mc_resolution: int
    texture_resolution: int
    model_save_format: ModelFormat


def process_image(
    model, img, out_dir: str, device: str, options: ProcessingOptions
) -> None:
    import torch

    with timed("Running model", device), torch.no_grad():
        scene_codes = model([img], device=device)

    if options.render:
        render_views(
            model, scene_codes, out_dir, options.render_views, options.render_fps
        )

    with timed("Extracting mesh", device):
        meshes = model.extract_mesh(
            scene_codes,
            not options.bake_texture,
            resolution=options.mc_resolution,
        )

    out_mesh_path = os.path.join(out_dir, f"mesh.{options.model_save_format.value}")
    if options.bake_texture:
        export_baked_mesh(
            model,
            scene_codes,
            meshes[0],
            out_dir,
            out_mesh_path,
            options.texture_resolution,
        )
    else:
        export_vertex_color_mesh(meshes[0], out_mesh_path)


def main(
    images: Annotated[
        list[Path],
        typer.Argument(
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to input image(s).",
        ),
    ],
    device: Annotated[
        str,
        typer.Option(
            help="Device to use. If the requested CUDA device is not available, falls back to 'mps' if present, otherwise 'cpu'."
        ),
    ] = "cuda:0",
    pretrained_model_name_or_path: Annotated[
        str,
        typer.Option(
            help="Path to the pretrained model. Could be either a huggingface model id or a local path."
        ),
    ] = "stabilityai/TripoSR",
    chunk_size: Annotated[
        int,
        typer.Option(
            min=0,
            help="Evaluation chunk size for surface extraction and rendering. Smaller chunk size reduces VRAM usage but increases computation time. 0 for no chunking.",
        ),
    ] = 8192,
    mc_resolution: Annotated[
        int, typer.Option(min=1, help="Marching cubes grid resolution.")
    ] = 256,
    no_remove_bg: Annotated[
        bool,
        typer.Option(
            "--no-remove-bg",
            help="Do NOT automatically remove the background. The input image should then be an RGB image with gray background and properly-sized foreground.",
        ),
    ] = False,
    foreground_ratio: Annotated[
        float,
        typer.Option(
            min=0.0,
            max=1.0,
            help="Ratio of the foreground size to the image size. Only used when --no-remove-bg is not specified.",
        ),
    ] = 0.85,
    output_dir: Annotated[
        Path, typer.Option(help="Output directory to save the results.")
    ] = Path("output/"),
    model_save_format: Annotated[
        ModelFormat, typer.Option(help="Format to save the extracted mesh.")
    ] = ModelFormat.obj,
    bake_texture: Annotated[
        bool,
        typer.Option(
            "--bake-texture",
            help="Bake a texture atlas for the extracted mesh, instead of vertex colors.",
        ),
    ] = False,
    texture_resolution: Annotated[
        int,
        typer.Option(
            min=1,
            help="Texture atlas resolution, only useful with --bake-texture.",
        ),
    ] = 2048,
    render: Annotated[
        bool, typer.Option("--render", help="Save a NeRF-rendered video.")
    ] = False,
    render_views: Annotated[
        int,
        typer.Option(min=1, help="Number of views to render. Only used with --render."),
    ] = 30,
    render_fps: Annotated[
        int,
        typer.Option(
            min=1, help="Frame rate of the rendered video. Only used with --render."
        ),
    ] = 30,
) -> None:
    """Reconstruct 3D meshes from single images with TripoSR."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO
    )

    os.makedirs(output_dir, exist_ok=True)

    device = resolve_device(device)

    model = load_model(pretrained_model_name_or_path, chunk_size, device)
    processed = preprocess_images(images, output_dir, no_remove_bg, foreground_ratio)

    options = ProcessingOptions(
        render=render,
        render_views=render_views,
        render_fps=render_fps,
        bake_texture=bake_texture,
        mc_resolution=mc_resolution,
        texture_resolution=texture_resolution,
        model_save_format=model_save_format,
    )
    for i, img in enumerate(processed):
        logging.info(f"Running image {i + 1}/{len(processed)} ...")
        out_dir = os.path.join(output_dir, str(i))
        process_image(model, img, out_dir, device, options)


def cli() -> None:
    """Console-script entry point (`triposr`)."""
    typer.run(main)


if __name__ == "__main__":
    cli()
