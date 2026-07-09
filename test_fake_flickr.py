"""Evaluate NPR on the FakeFlickr dataset using the Flickr30k test split.

NPR (Neighboring Pixel Relationships) is a lightweight modified ResNet-50 classifier
that detects the up-sampling artifacts left by generative models. The NPR feature
extraction (``x - interpolate(x, 0.5)``) lives *inside* the model's ``forward``, so we
simply feed ImageNet-normalized RGB tensors and take ``sigmoid()`` of the single logit.
As a binary classifier it has a natural 0.5 threshold, so -- unlike AEROBLADE -- no
global threshold calibration is needed.

For every generator under ``<dataset_root>/generated/<gen>/img`` this script:

  1. Filters images to the IDs listed in the Flickr30k test split.
  2. Removes the format/resolution confound (reals are JPEG, fakes are PNG/WebP at
     varying resolutions): by default it re-encodes every real and fake image to JPEG
     q90 in memory and center-crops at native resolution (no downscale). Real images
     come from ``<dataset_root>/real`` (or ``real_rescaled`` for the
     ``flux_fill_real_rescaled`` generator, which was conditioned on the rescaled reals).
     Reals are scored once per source and shared across generators.
  3. Runs the NPR classifier and reports ACC / AP / R_ACC / F_ACC per generator.

Results are written as one CSV row per generator.
"""

from __future__ import annotations

import argparse
import csv
import sys
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, average_precision_score
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from networks.resnet import resnet50  # noqa: E402

DEFAULT_GENERATORS = [
    "sd_1_5",
    "sd_3_5_large",
    "sdxl_turbo",
    "z_image_turbo",
    "flux_1_dev",
    "flux_fill_flux_1_dev",
    "flux_fill_sd_3_5_large",
    "flux_fill_real_rescaled",
]

# Generators that were conditioned on the rescaled-real source images.
# For these, the matching "real" is the rescaled PNG, not the original JPG.
RESCALED_REAL_GENS = {"flux_fill_real_rescaled"}

# NPR feeds RGB normalized with the standard ImageNet statistics.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def read_test_ids(split_file: Path) -> list[str]:
    with split_file.open("r") as f:
        ids = [line.strip() for line in f if line.strip()]
    if not ids:
        raise RuntimeError(f"Test split is empty: {split_file}")
    return ids


def find_image(dirpath: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPEG"):
        cand = dirpath / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


def jpeg_reencode(img: Image.Image, quality: int) -> Image.Image:
    """Round-trip a PIL image through JPEG in memory (format equalization).

    Mirrors ``data/datasets.py::pil_jpg``: making every real and fake image share the
    same JPEG-q90 compression removes the real-JPEG vs fake-PNG/WebP format confound.
    """
    buf = BytesIO()
    img.save(buf, format="jpeg", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def build_transform(crop_mode: str, load_size: int, crop_size: int) -> transforms.Compose:
    """NPR test transform.

    - ``crop`` (default): center-crop at native resolution -- no downscale, so no extra
      resampling confound is introduced. NPR needs even H/W (its internal
      ``interpolate(scale=0.5)``); a fixed even ``crop_size`` guarantees that.
    - ``resize``: short-side-agnostic ``Resize((load_size, load_size))`` then center crop
      (NPR-native ForenSynths style).
    """
    ops: list = []
    if crop_mode == "resize":
        ops.append(transforms.Resize((load_size, load_size)))
    ops.append(transforms.CenterCrop(crop_size))
    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(ops)


def classify_dir(
    model: torch.nn.Module,
    src_dir: Path,
    test_ids: list[str],
    label: int,
    transform: transforms.Compose,
    jpeg_equalize: bool,
    jpeg_quality: int,
    batch_size: int,
    device: torch.device,
    desc: str,
) -> tuple[list[float], list[int]]:
    """Score every test-split image found in ``src_dir`` with NPR.

    Returns ``(probs, labels)`` where each prob is ``sigmoid(logit)`` and each label is
    the given ``label`` (0 real / 1 fake). Images are read straight from the source dir --
    no disk staging.
    """
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Source folder not found: {src_dir}")

    probs: list[float] = []
    labels: list[int] = []
    batch: list[torch.Tensor] = []

    def flush(batch_tensors: list[torch.Tensor]) -> None:
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            out = model(x).sigmoid().flatten().cpu().numpy().tolist()
        probs.extend(out)
        labels.extend([label] * len(out))

    n = 0
    for img_id in tqdm(test_ids, desc=desc, dynamic_ncols=True, leave=False):
        src = find_image(src_dir, img_id)
        if src is None:
            continue
        try:
            img = Image.open(src).convert("RGB")
            if jpeg_equalize:
                img = jpeg_reencode(img, jpeg_quality)
        except Exception as exc:  # corrupt / unreadable
            print(f"  skip {src}: {exc}")
            continue
        batch.append(transform(img))
        n += 1
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
    flush(batch)
    if n == 0:
        raise RuntimeError(f"No test-split images matched under {src_dir}")
    return probs, labels


def evaluate_generator(
    real_probs: list[float],
    real_labels: list[int],
    fake_probs: list[float],
    fake_labels: list[int],
) -> dict[str, float]:
    y_true = np.array(real_labels + fake_labels)
    y_pred = np.array(real_probs + fake_probs)
    return {
        "ACC": float(accuracy_score(y_true, y_pred > 0.5)),
        "AP": float(average_precision_score(y_true, y_pred)),
        "R_ACC": float(accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5)),
        "F_ACC": float(accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5)),
        "N_real": int((y_true == 0).sum()),
        "N_fake": int((y_true == 1).sum()),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset-root", type=Path,
                   default=Path("/home/marek/FakeFlickr/data/fake-flickr"),
                   help="Root of the fake-flickr dataset.")
    p.add_argument("--test-split", type=Path,
                   default=Path("/home/marek/FakeFlickr/data/flickr30k_entities/test.txt"),
                   help="File with one Flickr30k image ID per line (the test split).")
    p.add_argument("--ckpt", type=Path, default=REPO_ROOT / "NPR.pth",
                   help="Path to the trained NPR classifier checkpoint (.pth).")
    p.add_argument("--generators", nargs="+", default=DEFAULT_GENERATORS,
                   help=f"Generator subdirs to evaluate (default: {DEFAULT_GENERATORS}).")
    p.add_argument("--results-csv", type=Path,
                   default=REPO_ROOT / "data" / "results" / "fake_flickr_npr.csv",
                   help="Output CSV path.")
    p.add_argument("--crop-mode", choices=["crop", "resize"], default="crop",
                   help="How to reach a fixed input size: 'crop' (native-res center crop, "
                        "no downscale -- default, removes the resolution confound without "
                        "adding resampling) or 'resize' (Resize to load-size then crop, "
                        "NPR-native ForenSynths style).")
    p.add_argument("--load-size", type=int, default=256,
                   help="Resize target for --crop-mode resize (ignored for 'crop').")
    p.add_argument("--crop-size", type=int, default=256,
                   help="Center-crop size (must be even for NPR).")
    p.add_argument("--no-jpeg-equalize", dest="jpeg_equalize", action="store_false",
                   default=True,
                   help="Disable JPEG re-encoding of inputs. Default ON to match the "
                        "FakeFlickr eval protocol (removes the real-JPEG vs fake-PNG/WebP "
                        "format confound).")
    p.add_argument("--jpeg-quality", type=int, default=90,
                   help="JPEG quality for the format-equalization round trip.")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Batch size for the classifier.")
    p.add_argument("--debug", action="store_true",
                   help="Debug mode: only run on --debug-samples images per set "
                        "to smoke-test the pipeline.")
    p.add_argument("--debug-samples", type=int, default=10,
                   help="Number of test IDs to keep in --debug mode.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.ckpt.is_file():
        raise FileNotFoundError(f"--ckpt not found: {args.ckpt}")
    if not args.test_split.is_file():
        raise FileNotFoundError(f"--test-split not found: {args.test_split}")
    if args.crop_size % 2 != 0:
        raise ValueError(f"--crop-size must be even for NPR, got {args.crop_size}")

    test_ids = read_test_ids(args.test_split)
    print(f"Loaded {len(test_ids)} test IDs from {args.test_split}")
    if args.debug:
        test_ids = test_ids[: args.debug_samples]
        print(f"[DEBUG] truncated to {len(test_ids)} IDs")

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading NPR classifier from {args.ckpt}")
    model = resnet50(num_classes=1)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    # NPR.pth is DataParallel-wrapped: strip the leading "module." from every key.
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    transform = build_transform(args.crop_mode, args.load_size, args.crop_size)
    print(f"Preprocess: crop_mode={args.crop_mode} crop_size={args.crop_size} "
          f"jpeg_equalize={args.jpeg_equalize} (q{args.jpeg_quality})")

    # Score reals once per source and reuse across generators that share it.
    real_sources = {("real_rescaled" if g in RESCALED_REAL_GENS else "real")
                    for g in args.generators}
    real_cache: dict[str, tuple[list[float], list[int]]] = {}
    for rn in sorted(real_sources):
        print(f"\n=== scoring shared reals: {rn} ===")
        real_cache[rn] = classify_dir(
            model, args.dataset_root / rn, test_ids, 0, transform,
            args.jpeg_equalize, args.jpeg_quality, args.batch_size, device,
            desc=f"  reals {rn}",
        )
        print(f"  scored {len(real_cache[rn][0])} real images ({rn})")

    rows: list[dict] = []
    for gen in args.generators:
        print(f"\n=== {gen} ===")
        rn = "real_rescaled" if gen in RESCALED_REAL_GENS else "real"
        fake_probs, fake_labels = classify_dir(
            model, args.dataset_root / "generated" / gen / "img", test_ids, 1, transform,
            args.jpeg_equalize, args.jpeg_quality, args.batch_size, device,
            desc=f"  fakes {gen}",
        )
        print(f"  scored {len(fake_probs)} fake images")

        real_probs, real_labels = real_cache[rn]
        metrics = evaluate_generator(real_probs, real_labels, fake_probs, fake_labels)
        print(f"  {gen}: " + " ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        ))
        rows.append({"generator": gen, **metrics})

    fieldnames = ["generator", "ACC", "AP", "R_ACC", "F_ACC", "N_real", "N_fake"]
    with args.results_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {args.results_csv}")


if __name__ == "__main__":
    main()
