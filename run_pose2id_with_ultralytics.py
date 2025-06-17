import argparse
import os
import tempfile
import urllib.parse
import urllib.request
from types import SimpleNamespace

from PIL import Image
try:
    from ultralytics import YOLO
except ImportError:  # allow running with --mock without the dependency
    YOLO = None

from ultralytics_tracker_adapter import (
    results_to_pose2id_inputs,
    generate_mock_results,
)


REQUIRED_WEIGHTS = [
    "denoising_unet.pth",
    "reference_unet.pth",
    "IFR.pth",
    "pose_guider.pth",
    "transformer_20.pth",
]


def ensure_pretrained_weights(path: str) -> None:
    """Download Pose2ID weights from HuggingFace if missing."""
    missing = [w for w in REQUIRED_WEIGHTS if not os.path.exists(os.path.join(path, w))]
    if not missing:
        return
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # network or package unavailable
        raise RuntimeError(
            f"Missing weights {missing} in {path}. Install `huggingface_hub` and "
            "ensure internet access to download them automatically."
        ) from exc

    snapshot_download("yuanc3/Pose2ID", local_dir=path, local_dir_use_symlinks=False)


def ensure_local_video(path: str) -> str:
    """Download remote video if the given path is an URL."""
    parsed = urllib.parse.urlparse(path)
    if parsed.scheme in {"http", "https"}:
        _, ext = os.path.splitext(parsed.path)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        urllib.request.urlretrieve(path, tmp.name)
        return tmp.name
    return path


def main(
    video_path: str,
    model_path: str,
    ckpt_dir: str,
    pose_dir: str,
    out_dir: str,
    config: str,
    mock: bool = False,
):
    ensure_pretrained_weights(ckpt_dir)
    if mock:
        results = generate_mock_results()
    else:
        if YOLO is None:
            raise ImportError(
                "ultralytics package is required unless running with --mock"
            )
        local_video = ensure_local_video(video_path)
        yolo = YOLO(model_path)
        results = list(yolo.track(source=local_video, persist=True))

    tracks = results_to_pose2id_inputs(results)
    # keep first frame per track id
    unique_tracks = {}
    for item in tracks:
        tid = item["id"]
        if tid not in unique_tracks:
            unique_tracks[tid] = item

    with tempfile.TemporaryDirectory() as tmpdir:
        for tid, item in unique_tracks.items():
            x1, y1, x2, y2 = map(int, item["bbox"])
            frame = item["frame"]
            crop = frame[y1:y2, x1:x2]
            # YOLO outputs BGR images; convert to RGB for PIL without cv2
            pil_img = Image.fromarray(crop[:, :, ::-1])
            pil_img.save(os.path.join(tmpdir, f"{tid}.jpg"))

        try:
            import IPG.inference as pose2id_inference
        except ImportError as exc:
            raise ImportError(
                "Pose2ID inference dependencies are required to run the full pipeline"
            ) from exc
        from omegaconf import OmegaConf

        if config.endswith(".yaml"):
            cfg = OmegaConf.load(config)
        elif config.endswith(".py"):
            cfg = pose2id_inference.import_filename(config).cfg
        else:
            raise ValueError(f"Unsupported config format: {config}")

        pose2id_inference.args = SimpleNamespace(
            ckpt_dir=ckpt_dir,
            pose_dir=pose_dir,
            ref_dir=tmpdir,
            out_dir=out_dir,
            config=config,
        )

        pose2id_inference.main(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Pose2ID with Ultralytics tracker")
    parser.add_argument("video", help="path to video")
    parser.add_argument("model", help="path to YOLO model or name e.g. yolov8n.pt")
    parser.add_argument("--ckpt_dir", default="pretrained")
    parser.add_argument("--pose_dir", default="standard_poses")
    parser.add_argument("--out_dir", default="output")
    parser.add_argument("--config", default=os.path.join("configs", "inference.yaml"))
    parser.add_argument("--mock", action="store_true", help="use randomly generated tracker data")
    args = parser.parse_args()
    main(
        args.video,
        args.model,
        args.ckpt_dir,
        args.pose_dir,
        args.out_dir,
        args.config,
        args.mock,
    )
