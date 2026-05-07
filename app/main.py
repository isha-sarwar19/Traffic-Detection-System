"""
app/main.py
===========
FastAPI web server for vehicle detection.

Endpoints:
  GET  /              -> Web UI
  GET  /health        -> Health check
  GET  /classes       -> List detectable classes
  POST /detect/image  -> Upload image, get detections + annotated image
  POST /detect/video  -> Upload video, get annotated video + stats
  GET  /docs          -> Swagger UI (auto-generated API docs)

Run:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import uuid
import base64
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT         = Path(__file__).parent.parent
STATIC_DIR   = ROOT / "static"
OUTPUTS_DIR  = STATIC_DIR / "outputs"
WEIGHTS_PATH = str(ROOT / "weights" / "yolov8n.pt")
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title       = "Vehicle Detection API",
    description = "YOLOv8 vehicle detection: car, truck, bus, motorcycle, bicycle.",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Lazy-load the model on first request
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        from app.model import VehicleDetector
        _detector = VehicleDetector(weights_path=WEIGHTS_PATH)
    return _detector


# ── Response Models ─────────────────────────────────────────────────────────

class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

class DetectionItem(BaseModel):
    class_id:   int
    class_name: str
    confidence: float
    bbox:       BBox

class DetectionResponse(BaseModel):
    success:             bool
    total_detections:    int
    class_counts:        dict
    inference_time_ms:   float
    image_shape:         List[int]
    detections:          List[DetectionItem]
    annotated_image_b64: Optional[str] = None
    output_url:          Optional[str] = None


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html = Path(__file__).parent / "templates" / "index.html"
    if html.exists():
        return HTMLResponse(content=html.read_text())
    return HTMLResponse("<h1>Vehicle Detection API</h1><a href='/docs'>Docs</a>")


@app.get("/health")
async def health():
    return {"status": "ok", "model": "YOLOv8s Vehicle Detection"}


@app.get("/classes")
async def get_classes():
    from app.model import CLASS_NAMES, CLASS_COLORS
    return {
        "total":   len(CLASS_NAMES),
        "classes": [
            {"id": i, "name": n, "color": list(CLASS_COLORS.get(n, [255, 255, 255]))}
            for i, n in enumerate(CLASS_NAMES)
        ],
    }


@app.post("/detect/image", response_model=DetectionResponse)
async def detect_image(
    file:         UploadFile = File(...),
    confidence:   float = Query(default=0.40, ge=0.10, le=0.95),
    return_image: bool  = Query(default=True),
    save_output:  bool  = Query(default=True),
):
    """Upload an image and get vehicle detections."""

    # Validate file type
    allowed = {"image/jpeg", "image/png", "image/bmp", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")

    # Read bytes and decode image
    raw = await file.read()
    if len(raw) > 50 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 50MB)")

    arr = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")

    # Run detection
    try:
        det           = get_detector()
        det.conf      = confidence
        result        = det.detect(img, draw=True)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"Detection error: {e}")

    # Build response
    items = [
        DetectionItem(
            class_id   = d.class_id,
            class_name = d.class_name,
            confidence = round(d.confidence, 4),
            bbox       = BBox(x1=d.bbox[0], y1=d.bbox[1], x2=d.bbox[2], y2=d.bbox[3]),
        )
        for d in result.detections
    ]

    output_url = None
    if save_output and result.annotated_image is not None:
        fname    = f"{uuid.uuid4().hex[:8]}_{file.filename}"
        out_path = OUTPUTS_DIR / fname
        cv2.imwrite(str(out_path), result.annotated_image)
        output_url = f"/static/outputs/{fname}"

    b64 = None
    if return_image and result.annotated_image is not None:
        _, buf = cv2.imencode(".jpg", result.annotated_image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64    = base64.b64encode(buf).decode()

    s = result.summary
    return DetectionResponse(
        success             = True,
        total_detections    = s["total_detections"],
        class_counts        = s["class_counts"],
        inference_time_ms   = s["inference_time_ms"],
        image_shape         = s["image_shape"],
        detections          = items,
        annotated_image_b64 = b64,
        output_url          = output_url,
    )


@app.post("/detect/video")
async def detect_video(
    file:       UploadFile = File(...),
    confidence: float = Query(default=0.40, ge=0.10, le=0.95),
    max_frames: int   = Query(default=150),
):
    """Upload a video and get an annotated output video + per-frame stats + analytics."""
    allowed = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"}
    if file.content_type not in allowed:
        raise HTTPException(400, "Use MP4/AVI/MOV format")

    uid    = uuid.uuid4().hex[:8]
    f_in   = OUTPUTS_DIR / f"in_{uid}.mp4"
    f_raw  = OUTPUTS_DIR / f"raw_{uid}.mp4"
    f_out  = OUTPUTS_DIR / f"out_{uid}.mp4"

    try:
        with open(f_in, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Read video FPS for timeline calculations
        cap = cv2.VideoCapture(str(f_in))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        det      = get_detector()
        det.conf = confidence
        results  = det.detect_video(str(f_in), output_path=str(f_raw), max_frames=max_frames)

        # Convert to H.264 for web browser compatibility
        ffmpeg_cmd = "/home/isha.sarwar@vaival.tech/miniconda3/lib/python3.13/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2"
        try:
            subprocess.run(
                [ffmpeg_cmd, "-y", "-i", str(f_raw), "-vcodec", "libx264", str(f_out)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True
            )
        except Exception as e:
            print(f"FFmpeg conversion failed: {e}")
            # Fallback to the raw video if compression fails
            shutil.copy(str(f_raw), str(f_out))
            
        if f_raw.exists():
            f_raw.unlink()

        total     = sum(len(r.detections) for r in results)
        avg_t     = sum(r.inference_time_ms for r in results) / max(len(results), 1)
        cls_total: dict = {}
        for r in results:
            for cls, cnt in r.summary["class_counts"].items():
                cls_total[cls] = cls_total.get(cls, 0) + cnt

        # ── Per-frame analytics ─────────────────────────────────────────
        def congestion_level(vehicle_count: int) -> str:
            if vehicle_count >= 20:
                return "HIGH"
            elif vehicle_count >= 10:
                return "MODERATE"
            return "LOW"

        frame_data = []
        peak_frame_idx = 0
        peak_vehicle_count = 0

        for i, r in enumerate(results):
            n_vehicles = len(r.detections)
            frame_class_counts = r.summary["class_counts"]
            level = congestion_level(n_vehicles)
            time_sec = round(i / video_fps, 2)

            frame_data.append({
                "frame":           i,
                "time_sec":        time_sec,
                "vehicle_count":   n_vehicles,
                "class_counts":    frame_class_counts,
                "congestion":      level,
            })

            if n_vehicles > peak_vehicle_count:
                peak_vehicle_count = n_vehicles
                peak_frame_idx = i

        # ── Congestion timeline (grouped by level changes) ──────────────
        congestion_timeline = []
        if frame_data:
            cur_level  = frame_data[0]["congestion"]
            start_time = frame_data[0]["time_sec"]
            start_frame = 0

            for fd in frame_data[1:]:
                if fd["congestion"] != cur_level:
                    congestion_timeline.append({
                        "level":       cur_level,
                        "start_sec":   start_time,
                        "end_sec":     fd["time_sec"],
                        "start_frame": start_frame,
                        "end_frame":   fd["frame"] - 1,
                    })
                    cur_level   = fd["congestion"]
                    start_time  = fd["time_sec"]
                    start_frame = fd["frame"]

            # Close last segment
            congestion_timeline.append({
                "level":       cur_level,
                "start_sec":   start_time,
                "end_sec":     frame_data[-1]["time_sec"],
                "start_frame": start_frame,
                "end_frame":   frame_data[-1]["frame"],
            })

        # ── Overall congestion ──────────────────────────────────────────
        avg_vehicles = total / max(len(results), 1)
        overall_congestion = congestion_level(int(avg_vehicles))

        # ── Peak traffic moment ─────────────────────────────────────────
        peak_info = {
            "frame":         peak_frame_idx,
            "time_sec":      round(peak_frame_idx / video_fps, 2),
            "vehicle_count": peak_vehicle_count,
            "congestion":    congestion_level(peak_vehicle_count),
            "class_counts":  frame_data[peak_frame_idx]["class_counts"] if frame_data else {},
        }

        return JSONResponse({
            "success":                  True,
            "frames_processed":         len(results),
            "total_detections":         total,
            "avg_detections_per_frame": round(avg_vehicles, 2),
            "avg_inference_ms":         round(avg_t, 2),
            "class_totals":             cls_total,
            "output_url":               f"/static/outputs/out_{uid}.mp4",
            # ── Analytics extensions ──
            "video_fps":                round(video_fps, 2),
            "frame_data":               frame_data,
            "congestion_timeline":      congestion_timeline,
            "overall_congestion":       overall_congestion,
            "peak_traffic":             peak_info,
            "vehicle_type_distribution": cls_total,  # alias for convenience
        })
    finally:
        if f_in.exists():
            f_in.unlink()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
