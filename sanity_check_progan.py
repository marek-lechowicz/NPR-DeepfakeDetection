"""Sanity-check the NPR environment/checkpoint against its native ProGAN test set.

NPR's own paper reports ~99%+ accuracy on the CNNDetection ProGAN test set (the
data it was trained on). This script runs NPR's own `TestOptions` /
`create_dataloader` / `validate` code -- unmodified -- against that test set to
confirm the checkpoint, architecture, and eval pipeline are all working correctly.
It exists to explain (and rule out as a bug) the much lower accuracy NPR gets
on the FakeFlickr benchmark (see `test_fake_flickr.py` / `data/results/fake_flickr_*.csv`
in this repo): if NPR still gets ~99% here, the FakeFlickr result is a genuine
GAN-to-diffusion generalization gap, not a broken harness.

Data: see ../progan_sanity_check/README.md for where progan_testset/ comes from.

Run with an interpreter that has torch+cuda, torchvision, opencv-python, scikit-learn
(e.g. `DIRE/.venv/bin/python`, which already has all of these):

    CUDA_VISIBLE_DEVICES=0 /path/to/venv/bin/python sanity_check_progan.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from networks.resnet import resnet50  # noqa: E402
from options.test_options import TestOptions  # noqa: E402
from validate import validate  # noqa: E402

DATA_ROOT = REPO_ROOT.parent / "progan_sanity_check" / "progan_testset" / "progan"
CKPT = REPO_ROOT / "NPR.pth"
RESULTS_CSV = REPO_ROOT / "data" / "results" / "progan_sanity_npr.csv"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required (validate() moves batches to .cuda()).")
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(
            f"ProGAN test set not found at {DATA_ROOT}. See "
            f"{REPO_ROOT.parent / 'progan_sanity_check' / 'README.md'} for how to get it.")

    classes = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir())
    print(f"Found {len(classes)} ProGAN test categories: {classes}")

    model = resnet50(num_classes=1)
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    # NPR.pth is DataParallel-wrapped: strip the leading "module." from every key.
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.cuda().eval()
    print("Checkpoint loaded OK (strict=True).")

    rows = []
    all_true: list[float] = []
    all_pred: list[float] = []
    for cls in classes:
        sys.argv = [
            "validate.py",
            "--dataroot", str(DATA_ROOT / cls),
            "--classes", "",
            "--no_crop",
            "--batch_size", "32",
            "--num_threads", "4",
            "--model_path", str(CKPT),
        ]
        opt = TestOptions().parse(print_options=False)
        acc, ap, r_acc, f_acc, y_true, y_pred = validate(model, opt)
        print(f"  {cls:14s} ACC={acc:.4f} AP={ap:.4f} R_ACC={r_acc:.4f} F_ACC={f_acc:.4f} n={len(y_true)}")
        rows.append({"category": cls, "ACC": acc, "AP": ap, "R_ACC": r_acc, "F_ACC": f_acc,
                      "N_real": int((y_true == 0).sum()), "N_fake": int((y_true == 1).sum())})
        all_true.extend(y_true.tolist())
        all_pred.extend(y_pred.tolist())

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    pooled = {
        "category": "POOLED",
        "ACC": accuracy_score(y_true, y_pred > 0.5),
        "AP": average_precision_score(y_true, y_pred),
        "R_ACC": accuracy_score(y_true[y_true == 0], y_pred[y_true == 0] > 0.5),
        "F_ACC": accuracy_score(y_true[y_true == 1], y_pred[y_true == 1] > 0.5),
        "N_real": int((y_true == 0).sum()),
        "N_fake": int((y_true == 1).sum()),
    }
    rows.append(pooled)
    print("\n=== POOLED across all 20 ProGAN categories ===")
    for k, v in pooled.items():
        print(f"  {k}: {v}")

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "ACC", "AP", "R_ACC", "F_ACC", "N_real", "N_fake"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {RESULTS_CSV}")


if __name__ == "__main__":
    main()
