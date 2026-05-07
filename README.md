<div align="center">

# 🚗 VehicleEye

### YOLOv8 Vehicle Detection & Traffic Analytics System

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFAA?style=for-the-badge)](https://docs.ultralytics.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

**Real-time vehicle detection, classification, and traffic flow analytics powered by a custom-trained YOLOv8s model achieving 97.8% mAP50.**

[Features](#-features) · [Demo](#-demo) · [Installation](#-installation) · [Usage](#-usage) · [API](#-api-reference) · [Model](#-model-performance)

</div>

---

## 📌 Overview

**VehicleEye** is a computer vision system that detects and classifies vehicles in both images and videos using a custom-trained YOLOv8s model. It features a full-stack web application with a dark-themed dashboard that provides real-time traffic analytics, congestion monitoring, and downloadable annotated outputs — all served through a lightning-fast FastAPI backend.

> Built for traffic engineers, smart city researchers, and CV enthusiasts who need accurate, interpretable vehicle detection with zero cloud dependency.

---

## 🖥️ Demo

|                                           Image Detection                                            |                                                        Video Analytics Dashboard                                                        |
| :--------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------: |
| Upload any traffic image and get annotated bounding boxes with per-class counts and a live bar chart | Upload a pre-recorded traffic video and receive a full analytics report including flow graphs, congestion timelines, and peak detection |

**Dashboard Highlights:**

- 🎯 Annotated output image/video with confidence scores
- 📊 Per-class detection bar chart (car, truck, bus, motorcycle, bicycle)
- 📈 Traffic flow line graph (vehicles per second over time)
- 🔴 Congestion timeline with LOW / MODERATE / HIGH levels
- ⏱️ Inference time display and peak traffic moment detection

---

## ✨ Features

### 🖼️ Image Detection

- 📂 Drag & drop image upload interface
- 🎚️ Adjustable confidence threshold slider
- 🖍️ Returns annotated image with bounding boxes and class labels
- 📊 Per-class detection count bar chart (powered by Chart.js)
- ⚡ Real-time inference time display

### 🎬 Video Analytics Dashboard

- 📹 Upload pre-recorded traffic videos for frame-by-frame analysis
- 📈 Traffic flow line graph — vehicles detected per second
- 🚦 Congestion level classification:
  - 🟢 **LOW** — fewer than 10 vehicles
  - 🟡 **MODERATE** — 10 to 19 vehicles
  - 🔴 **HIGH** — 20 or more vehicles
- 🏆 Peak traffic moment detection
- 📥 Downloadable annotated output video
- 🥧 Vehicle type distribution chart

### 🔌 REST API

- Full OpenAPI/Swagger documentation at `/docs`
- JSON responses with detection metadata
- Health check and class listing endpoints
- Suitable for integration with CCTV pipelines or IoT dashboards

---

## 📊 Model Performance

Trained for **50 epochs** on **1,452 images** using the YOLOv8s architecture.

| Metric        | Score     |
| ------------- | --------- |
| **mAP50**     | **97.8%** |
| **mAP50-95**  | **88.4%** |
| **Precision** | **98.3%** |
| **Recall**    | **93.9%** |

### Per-Class mAP50

| Class         | mAP50 |
| ------------- | ----- |
| 🚌 Bus        | 99.2% |
| 🚛 Truck      | 98.0% |
| 🏍️ Motorcycle | 97.2% |
| 🚗 Car        | 96.7% |

---

## 🗂️ Project Structure

```
traffic-detection/
│
├── app/
│   ├── main.py               # FastAPI server (335 lines)
│   ├── model.py              # YOLOv8 inference engine
│   └── templates/
│       └── index.html        # Web UI — dark-themed dashboard
│
├── dataset/
│   ├── images/
│   │   ├── train/            # 1452 training images
│   │   ├── val/              # 401 validation images
│   │   └── test/             # 208 test images
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── prepare_dataset.py    # Dataset preparation script
│
├── weights/
│   └── best.pt               # Trained model weights (22 MB)
│
├── static/
│   └── outputs/              # Saved detection results
│
├── train.py                  # YOLOv8 training script
├── data.yaml                 # Dataset configuration
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

| Property       | Details                                                                                                        |
| -------------- | -------------------------------------------------------------------------------------------------------------- |
| **Source**     | [Kaggle — vehicle-dataset-for-yolo](https://www.kaggle.com/datasets/nadinpethiyagoda/vehicle-dataset-for-yolo) |
| **Train**      | 1,452 images                                                                                                   |
| **Validation** | 401 images                                                                                                     |
| **Test**       | 208 images                                                                                                     |
| **Classes**    | `car`, `truck`, `bus`, `motorcycle`, `bicycle`                                                                 |
| **Format**     | YOLOv8 (`.txt` labels, normalized coordinates)                                                                 |

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- Conda (recommended)
- 8GB+ RAM (16GB recommended for training)
- Linux / macOS / Windows

### Step 1 — Clone the Repository

```bash
git clone https://github.com/your-username/vehicleeye.git
cd vehicleeye
```

### Step 2 — Create Conda Environment

```bash
conda create -n traffic_detection python=3.10 -y
conda activate traffic_detection
```

### Step 3 — Install PyTorch (CPU)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Prepare the Dataset

Download the dataset from Kaggle and run:

```bash
python dataset/prepare_dataset.py --source /path/to/kaggle/dataset
```

### Step 6 — Train the Model _(optional — pretrained weights included)_

```bash
python train.py
```

> ✅ Pre-trained weights are already included at `weights/best.pt`. Skip training if you just want to run the app.

### Step 7 — Start the Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Step 8 — Open the Dashboard

```
http://localhost:8000
```

---

## 🚀 Usage

### Image Detection

1. Open `http://localhost:8000`
2. Drag & drop or click to upload a traffic image
3. Adjust the confidence threshold (default: 0.25)
4. Click **Detect** — view the annotated image and detection chart

### Video Analytics

1. Navigate to the **Video** tab on the dashboard
2. Upload a `.mp4` or `.avi` traffic video
3. Wait for processing — a full analytics report is generated
4. View the traffic flow graph, congestion timeline, and peak moment
5. Download the annotated video

### API (Programmatic)

**Detect vehicles in an image:**

```bash
curl -X POST "http://localhost:8000/detect/image" \
  -F "file=@traffic_image.jpg" \
  -F "confidence=0.25"
```

**Process a video:**

```bash
curl -X POST "http://localhost:8000/detect/video" \
  -F "file=@traffic_video.mp4"
```

**Check available classes:**

```bash
curl http://localhost:8000/classes
```

---

## 🧠 Training Details

| Parameter          | Value                                      |
| ------------------ | ------------------------------------------ |
| **Model**          | YOLOv8s                                    |
| **Epochs**         | 50                                         |
| **Batch Size**     | 8                                          |
| **Image Size**     | 416 × 416                                  |
| **Optimizer**      | AdamW                                      |
| **Device**         | CPU                                        |
| **Early Stopping** | Patience: 15 epochs                        |
| **Augmentations**  | Mosaic, MixUp, HSV shifts, Horizontal Flip |

---

## 🛠️ Tech Stack

| Layer               | Technology                |
| ------------------- | ------------------------- |
| **Detection Model** | YOLOv8s (Ultralytics)     |
| **Deep Learning**   | PyTorch (CPU)             |
| **Computer Vision** | OpenCV                    |
| **Backend**         | FastAPI (Python)          |
| **Frontend**        | HTML5 / CSS3 / Vanilla JS |
| **Charts**          | Chart.js                  |
| **Server**          | Uvicorn (ASGI)            |
| **Environment**     | Conda, Python 3.10        |

---

## 💻 Environment

| Property      | Spec                       |
| ------------- | -------------------------- |
| **OS**        | Linux                      |
| **Python**    | 3.10                       |
| **RAM**       | 16 GB                      |
| **CPU**       | Intel Core i7-8650U        |
| **Device**    | CPU only (no GPU required) |
| **Conda Env** | `traffic_detection`        |

---
