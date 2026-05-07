"""
app/model.py
============
Loads the trained YOLOv8 model and runs vehicle detection
on images or video frames.

Supported weight formats:
  - best.pt   (PyTorch, default)
  - best.onnx (ONNX, faster on CPU, use after exporting)
"""

import time
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

ROOT            = Path(__file__).parent.parent
DEFAULT_WEIGHTS = ROOT / "weights" / "best.pt"

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.40   # minimum confidence to count as a detection
IOU_THRESHOLD        = 0.45   # non-max suppression threshold

# Filter COCO classes: bicycle(1), car(2), motorcycle(3), bus(5), truck(7)
VEHICLE_CLASSES = [1, 2, 3, 5, 7]

# BGR colors for bounding boxes (one per class)
CLASS_COLORS = {
    "car":        (86,  180, 233),   # blue
    "truck":      (230, 159,   0),   # orange
    "bus":        (  0, 158, 115),   # green
    "motorcycle": (  0, 114, 178),   # dark blue
    "bicycle":    (240, 228,  66),   # yellow
}


@dataclass
class Detection:
    class_id:   int
    class_name: str
    confidence: float
    bbox:       Tuple[int, int, int, int]   # x1, y1, x2, y2 in pixels
    color:      Tuple[int, int, int] = field(default_factory=lambda: (255, 255, 255))


@dataclass
class InferenceResult:
    detections:        List[Detection]
    inference_time_ms: float
    image_shape:       Tuple[int, int]        # (height, width)
    annotated_image:   Optional[np.ndarray] = None

    @property
    def summary(self) -> dict:
        counts = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return {
            "total_detections":  len(self.detections),
            "class_counts":      counts,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "image_shape":       list(self.image_shape),
        }


class VehicleDetector:
    """
    YOLOv8 vehicle detector.

    Example:
        detector = VehicleDetector()
        result   = detector.detect(cv2.imread("traffic.jpg"))
        cv2.imwrite("output.jpg", result.annotated_image)
    """

    def __init__(
        self,
        weights_path: str = None,
        conf:  float = CONFIDENCE_THRESHOLD,
        iou:   float = IOU_THRESHOLD,
        imgsz: int   = 416,
    ):
        self.weights_path = weights_path or str(DEFAULT_WEIGHTS)
        self.conf  = conf
        self.iou   = iou
        self.imgsz = imgsz
        self.model = None
        self._load()

    def _load(self):
        """Load model weights into memory (runs once at startup)."""
        w = Path(self.weights_path)
        if not w.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.weights_path}\n"
                f"Train first: python train.py"
            )
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.weights_path)
            print(f"Model loaded: {w.name}")
        except ImportError:
            raise ImportError("Run: pip install ultralytics")

    def detect(self, image: np.ndarray, draw: bool = True, use_tracking: bool = False) -> InferenceResult:
        """
        Run detection on a single BGR image (numpy array).

        Args:
            image : OpenCV BGR image (from cv2.imread or camera frame)
            draw  : if True, returns annotated image with boxes drawn
            use_tracking: if True, uses ByteTrack algorithm for smooth detection across frames

        Returns:
            InferenceResult containing detections and optional annotated image
        """
        h, w = image.shape[:2]
        t0   = time.perf_counter()

        kwargs = dict(
            source  = image,
            conf    = self.conf,
            iou     = self.iou,
            imgsz   = self.imgsz,
            classes = VEHICLE_CLASSES,  # only detect vehicles
            verbose = False,
            device  = "cpu",
        )

        if use_tracking:
            raw = self.model.track(**kwargs, persist=True, tracker="bytetrack.yaml")
        else:
            raw = self.model.predict(**kwargs)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        detections = self._parse(raw[0])
        annotated  = self._draw(image.copy(), detections) if draw else None

        return InferenceResult(
            detections        = detections,
            inference_time_ms = elapsed_ms,
            image_shape       = (h, w),
            annotated_image   = annotated,
        )

    def detect_from_file(self, path: str, draw: bool = True) -> InferenceResult:
        """Load image from file path and run detection."""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        return self.detect(img, draw=draw)

    def _parse(self, result) -> List[Detection]:
        """Convert raw YOLOv8 output into Detection objects."""
        detections = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            cid  = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

            name  = result.names[cid]  # Map ID directly from model's internal classes

            color = CLASS_COLORS.get(name, (255, 255, 255))

            detections.append(Detection(
                class_id   = cid,
                class_name = name,
                confidence = conf,
                bbox       = (x1, y1, x2, y2),
                color      = color,
            ))
        return detections

    def _draw(self, image: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels on image."""
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            color = det.color
            label = f"{det.class_name} {det.confidence:.0%}"

            # Bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(image, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)

            # Label text
            cv2.putText(image, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        # Detection count overlay
        cv2.putText(image, f"Vehicles: {len(detections)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return image

    def detect_video(
        self,
        video_path:  str,
        output_path: str = None,
        max_frames:  int = None,
    ) -> List[InferenceResult]:
        """Process each frame of a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        fw  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (fw, fh))

        results       = []
        frame_count   = 0
        heatmap_accum = np.zeros((fh, fw), dtype=np.float32)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            r = self.detect(frame, draw=True, use_tracking=True)
            results.append(r)
            
            # ── Dynamic Heatmap Overlay ──
            # Add heat for every vehicle detected
            for det in r.detections:
                x1, y1, x2, y2 = det.bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(heatmap_accum, (cx, cy), 24, (4.0,), -1)
                
            # Render the overlay
            if writer and r.annotated_image is not None:
                # Soften the edges of the heat blobs
                heat_blur = cv2.GaussianBlur(heatmap_accum, (61, 61), 0)
                heat_norm = np.clip(heat_blur / 120.0, 0, 1.0) # 120 equates to max solid heat
                
                # Apply Turbo/Jet colormap for stunning thermal visuals
                colored_heat = cv2.applyColorMap(np.uint8(255 * heat_norm), cv2.COLORMAP_TURBO)
                
                # Dynamic blending mask: areas with high heat become more opaque (max 60%)
                alpha_mask = np.clip(heat_norm * 1.5, 0, 0.55)
                alpha_3d   = np.repeat(alpha_mask[:, :, np.newaxis], 3, axis=2)
                
                # Blend the heatmap precisely onto the original video
                r.annotated_image = (r.annotated_image * (1 - alpha_3d) + colored_heat * alpha_3d).astype(np.uint8)
                
                writer.write(r.annotated_image)
            
            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break

        cap.release()
        if writer:
            writer.release()

        print(f"Video processed: {frame_count} frames -> {output_path}")
        return results
