"""
prepare_dataset.py
==================
This script takes the downloaded Kaggle dataset and organizes it
into the correct folder structure for YOLOv8 training.

It handles two cases:
  1. Dataset already in YOLO format (images + .txt labels)
  2. Dataset needs class remapping to match our 5 classes

Our 5 target classes:
    0: car
    1: truck
    2: bus
    3: motorcycle
    4: bicycle

USAGE:
    python dataset/prepare_dataset.py --source /path/to/kaggle/download
"""

import os
import shutil
import random
import argparse
from pathlib import Path

# Project root
ROOT        = Path(__file__).parent.parent
DATASET_DIR = ROOT / "dataset"
IMAGES_DIR  = DATASET_DIR / "images"
LABELS_DIR  = DATASET_DIR / "labels"

# Our 5 target classes
OUR_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]

# ── Kaggle dataset: nadinpethiyagoda/vehicle-dataset-for-yolo ─────────────────
# Original classes: car(0), threewheel(1), bus(2), truck(3), motorbike(4), van(5)
# We remap to our 5 classes, dropping threewheel and van
KAGGLE_REMAP = {
    0: 0,    # car       → car (0)
    2: 2,    # bus       → bus (2)
    3: 1,    # truck     → truck (1)
    4: 3,    # motorbike → motorcycle (3)
    # 1 (threewheel) → skip
    # 5 (van)        → skip (or map to truck if you want)
}

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.20
TEST_RATIO  = 0.10


def setup_dirs():
    """Create train/val/test folders for images and labels."""
    for split in ["train", "val", "test"]:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)
    print("Folder structure created.")


def find_image_label_pairs(source_dir: Path):
    """
    Scan source directory recursively for image+label pairs.
    Returns list of (image_path, label_path) tuples.
    """
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    pairs = []

    for img_path in source_dir.rglob("*"):
        if img_path.suffix.lower() not in image_exts:
            continue

        # Look for matching label file
        # Try same folder first, then labels/ sibling folder
        label_candidates = [
            img_path.with_suffix(".txt"),
            img_path.parent.parent / "labels" / img_path.with_suffix(".txt").name,
            img_path.parent / "labels" / img_path.with_suffix(".txt").name,
        ]

        for lbl_path in label_candidates:
            if lbl_path.exists():
                pairs.append((img_path, lbl_path))
                break

    return pairs


def remap_label_file(src_label: Path, dst_label: Path, remap: dict) -> bool:
    """
    Read a YOLO label file, remap class IDs, write to destination.
    Returns True if at least one valid line was written, False otherwise.
    """
    lines = src_label.read_text().strip().split("\n")
    new_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue

        original_id = int(parts[0])
        if original_id not in remap:
            continue  # skip classes we don't want

        new_id    = remap[original_id]
        new_line  = f"{new_id} {' '.join(parts[1:])}"
        new_lines.append(new_line)

    if new_lines:
        dst_label.write_text("\n".join(new_lines))
        return True
    return False


def split_and_copy(pairs: list, remap: dict = None):
    """
    Shuffle pairs, split into train/val/test, copy files.
    Applies class remapping if remap dict is provided.
    """
    random.seed(42)
    random.shuffle(pairs)

    n       = len(pairs)
    n_train = int(n * TRAIN_RATIO)
    n_val   = int(n * VAL_RATIO)

    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train:n_train + n_val],
        "test":  pairs[n_train + n_val:],
    }

    total_copied = 0
    total_skipped = 0

    for split, split_pairs in splits.items():
        copied  = 0
        skipped = 0

        for img_path, lbl_path in split_pairs:
            dst_img = IMAGES_DIR / split / img_path.name
            dst_lbl = LABELS_DIR / split / (img_path.stem + ".txt")

            # Copy image
            shutil.copy2(img_path, dst_img)

            # Copy/remap label
            if remap:
                ok = remap_label_file(lbl_path, dst_lbl, remap)
                if ok:
                    copied += 1
                else:
                    # No valid classes after remapping, remove image too
                    dst_img.unlink(missing_ok=True)
                    skipped += 1
            else:
                shutil.copy2(lbl_path, dst_lbl)
                copied += 1

        print(f"  {split:6s}: {copied:4d} images copied | {skipped:3d} skipped (no target classes)")
        total_copied  += copied
        total_skipped += skipped

    return total_copied


def create_data_yaml():
    """Generate data.yaml for YOLOv8 training."""
    content = f"""# YOLOv8 Vehicle Detection Dataset Config
# Auto-generated by prepare_dataset.py

path: {ROOT.resolve()}
train: dataset/images/train
val:   dataset/images/val
test:  dataset/images/test

# Number of classes
nc: {len(OUR_CLASSES)}

# Class names (order = class ID)
names:
  0: car
  1: truck
  2: bus
  3: motorcycle
  4: bicycle
"""
    yaml_path = ROOT / "data.yaml"
    yaml_path.write_text(content)
    print(f"data.yaml written to: {yaml_path}")


def print_summary():
    print("\n" + "=" * 50)
    print("  DATASET SUMMARY")
    print("=" * 50)
    for split in ["train", "val", "test"]:
        imgs = len(list((IMAGES_DIR / split).glob("*.jpg"))) + \
               len(list((IMAGES_DIR / split).glob("*.png")))
        lbls = len(list((LABELS_DIR / split).glob("*.txt")))
        print(f"  {split:6s}  ->  {imgs:4d} images  |  {lbls:4d} labels")
    print("=" * 50)
    print("  Next step: python train.py")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=str, required=True,
        help="Path to extracted Kaggle dataset folder"
    )
    parser.add_argument(
        "--no-remap", action="store_true",
        help="Skip class remapping (use if dataset already has correct class IDs)"
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        print(f"ERROR: Source folder not found: {source}")
        exit(1)

    print(f"Source folder : {source}")
    print(f"Target classes: {OUR_CLASSES}")
    print("")

    setup_dirs()

    print("Scanning for image/label pairs...")
    pairs = find_image_label_pairs(source)
    print(f"Found {len(pairs)} image-label pairs")

    if len(pairs) == 0:
        print("ERROR: No image-label pairs found in the source folder.")
        print("Make sure the Kaggle dataset is extracted and contains both images and .txt label files.")
        exit(1)

    remap = None if args.no_remap else KAGGLE_REMAP

    print("\nSplitting and copying...")
    split_and_copy(pairs, remap=remap)

    create_data_yaml()
    print_summary()
