"""
train.py
========
Train YOLOv8s on the vehicle detection dataset.

Optimized for: Linux, CPU-only, 16GB RAM, ~500 images

USAGE:
    # Full training (~2-4 hours on CPU)
    python train.py

    # Quick test run (3 epochs, verify everything works)
    python train.py --test-run

    # Resume interrupted training
    python train.py --resume

    # Validate trained model
    python train.py --validate-only

    # Export best.pt to ONNX (faster CPU inference)
    python train.py --export-only
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT        = Path(__file__).parent
DATA_YAML   = ROOT / "data.yaml"
WEIGHTS_DIR = ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

# Training config optimized for your machine (Linux, 16GB RAM, CPU)
CONFIG = {
    "model_size":     "s",             # yolov8s: best accuracy/speed balance on CPU
    "epochs":         50,
    "batch":          8,               # safe for 8GB available RAM
    "imgsz":          416,             # good accuracy, faster than 640 on CPU
    "device":         "cpu",
    "optimizer":      "AdamW",         # better than SGD for small datasets
    "lr0":            0.001,
    "warmup_epochs":  3,
    "patience":       15,              # early stop if no improvement for 15 epochs
    "project":        "runs/train",
    "name":           "vehicle_yolov8s",
}


def check_requirements():
    """Verify ultralytics is installed."""
    try:
        import ultralytics
        print(f"ultralytics {ultralytics.__version__} found.")
    except ImportError:
        print("ERROR: ultralytics not installed.")
        print("Run: pip install ultralytics")
        sys.exit(1)


def check_dataset():
    """Verify dataset exists before training."""
    train_dir = ROOT / "dataset" / "images" / "train"
    if not train_dir.exists():
        print("ERROR: Training images not found.")
        print("Run first: python dataset/prepare_dataset.py --source /path/to/kaggle/download")
        sys.exit(1)

    imgs = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
    if len(imgs) == 0:
        print("ERROR: No images found in dataset/images/train/")
        print("Run: python dataset/prepare_dataset.py --source /path/to/kaggle/download")
        sys.exit(1)

    print(f"Dataset found: {len(imgs)} training images.")
    return len(imgs)


def train(test_run: bool = False, resume: bool = False):
    """Run YOLOv8 training."""
    from ultralytics import YOLO

    cfg = CONFIG.copy()

    if test_run:
        # Quick 3-epoch run to verify everything works before full training
        cfg["epochs"] = 3
        cfg["batch"]  = 4
        cfg["imgsz"]  = 320
        cfg["name"]   = "vehicle_test_run"
        print("TEST RUN MODE (3 epochs only - to verify setup)")

    model_file = f"yolov8{cfg['model_size']}.pt"

    print(f"""
  Training Config
  ---------------
  Model     : YOLOv8{cfg['model_size']} ({model_file})
  Epochs    : {cfg['epochs']}
  Batch     : {cfg['batch']}
  Img Size  : {cfg['imgsz']}
  Device    : {cfg['device']}
  Data      : {DATA_YAML}
""")

    # Load base model
    # yolov8s.pt downloads automatically (~22MB, pretrained on COCO)
    # Pretrained weights mean the model already knows basic visual features.
    # We fine-tune it specifically for vehicle detection.
    if resume:
        last_ckpt = Path(cfg["project"]) / cfg["name"] / "weights" / "last.pt"
        if last_ckpt.exists():
            print(f"Resuming from checkpoint: {last_ckpt}")
            model = YOLO(str(last_ckpt))
        else:
            print("No checkpoint found. Starting fresh.")
            model = YOLO(model_file)
    else:
        model = YOLO(model_file)

    # Start training
    results = model.train(
        data           = str(DATA_YAML),
        epochs         = cfg["epochs"],
        batch          = cfg["batch"],
        imgsz          = cfg["imgsz"],
        device         = cfg["device"],
        project        = cfg["project"],
        name           = cfg["name"],
        resume         = resume,

        # Optimizer
        optimizer      = cfg["optimizer"],
        lr0            = cfg["lr0"],
        warmup_epochs  = cfg["warmup_epochs"],
        patience       = cfg["patience"],

        # Augmentation
        # These transforms artificially increase dataset diversity
        augment        = True,
        hsv_h          = 0.015,    # hue shift ±1.5%
        hsv_s          = 0.7,      # saturation shift
        hsv_v          = 0.4,      # brightness shift
        fliplr         = 0.5,      # 50% horizontal flip
        mosaic         = 1.0,      # mosaic: combine 4 images into 1 (great for small datasets)
        mixup          = 0.1,      # mixup: blend 2 images (10% probability)
        degrees        = 5.0,      # random rotation ±5 degrees

        # Output
        save           = True,
        save_period    = 10,       # checkpoint every 10 epochs
        val            = True,
        verbose        = True,
        plots          = True,     # save training curve plots
    )

    # Copy best weights to /weights/
    best_src = Path(cfg["project"]) / cfg["name"] / "weights" / "best.pt"
    if best_src.exists():
        best_dst = WEIGHTS_DIR / "best.pt"
        shutil.copy(best_src, best_dst)
        print(f"\nBest model saved to: {best_dst}")

        if not test_run:
            export_onnx(str(best_dst), cfg["imgsz"])
    else:
        print("WARNING: best.pt not found after training.")

    return results


def export_onnx(weights_path: str, imgsz: int = 416):
    """
    Export trained PyTorch model to ONNX format.
    ONNX inference is ~30-50% faster on CPU than the .pt format.
    """
    from ultralytics import YOLO
    print("\nExporting to ONNX (faster CPU inference)...")

    model = YOLO(weights_path)
    path  = model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True)

    dst = WEIGHTS_DIR / "best.onnx"
    shutil.copy(path, dst)
    print(f"ONNX model saved to: {dst}")
    return str(dst)


def validate(weights_path: str = None):
    """
    Run validation on the val split and print accuracy metrics.

    Key metrics:
      mAP50    = detection accuracy at 50% IoU threshold (main metric)
      mAP50-95 = stricter accuracy (average across IoU 50%-95%)
      Precision = of all detected objects, how many are correct
      Recall    = of all real objects, how many were found
    """
    from ultralytics import YOLO

    weights_path = weights_path or str(WEIGHTS_DIR / "best.pt")
    if not Path(weights_path).exists():
        print(f"ERROR: Weights not found at {weights_path}")
        print("Train the model first: python train.py")
        return

    model   = YOLO(weights_path)
    metrics = model.val(data=str(DATA_YAML), device="cpu")

    print(f"""
  Validation Results
  ------------------
  mAP50      : {metrics.box.map50:.4f}
  mAP50-95   : {metrics.box.map:.4f}
  Precision  : {metrics.box.mp:.4f}
  Recall     : {metrics.box.mr:.4f}
""")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 vehicle detector")
    parser.add_argument("--test-run",      action="store_true", help="Quick 3-epoch test")
    parser.add_argument("--resume",        action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--validate-only", action="store_true", help="Only run validation")
    parser.add_argument("--export-only",   action="store_true", help="Export best.pt to ONNX")
    args = parser.parse_args()

    check_requirements()

    if args.validate_only:
        validate()
    elif args.export_only:
        export_onnx(str(WEIGHTS_DIR / "best.pt"))
    else:
        check_dataset()
        train(test_run=args.test_run, resume=args.resume)
        if not args.test_run:
            validate()
            print("\nDone! Start the web app with:")
            print("  uvicorn app.main:app --host 0.0.0.0 --port 8000")
