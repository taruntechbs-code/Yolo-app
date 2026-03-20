# 🦺 HWB-ZENITHRA
### AI-Powered Safety Equipment Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF5E00?style=for-the-badge&logo=yolo&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-brightgreen?style=for-the-badge)

> **An intelligent, real-time workplace safety monitoring system** that leverages YOLOv8 object detection to identify safety equipment (fire extinguishers) and classify environmental conditions (clutter) — deployable via webcam, image, video, or a Flask web interface.

---

## 📌 Table of Contents

1. [Overview](#-overview)
2. [Key Features](#-key-features)
3. [Project Structure](#-project-structure)
4. [Requirements](#-requirements)
5. [Installation](#-installation)
6. [Training the Model](#️-training-the-model)
7. [Running Inference](#-running-inference)
8. [Web Application](#-web-application)
9. [Performance Metrics](#-performance-metrics)
10. [Output Explanation](#-output-explanation)
11. [Advanced Usage](#-advanced-usage)
12. [Troubleshooting](#-troubleshooting)
13. [References & Citations](#-references--citations)
14. [License](#-license)

---

## 🔍 Overview

**HWB-ZENITHRA** is a deep learning-based object detection system designed for **workplace safety monitoring**. Built on the [YOLOv8](https://github.com/ultralytics/ultralytics) architecture, it detects critical safety equipment such as **fire extinguishers** and classifies environmental conditions like **clutter levels**, enabling proactive hazard identification in industrial, commercial, and institutional environments.

The system offers:

- A **training pipeline** (baseline + optimized models)
- **Multiple inference modes** (image, video, real-time webcam)
- A **Flask-based web app** for browser-accessible predictions

This project was developed as part of a hackathon-grade submission (SIH-level) and follows production-ready ML engineering practices.

---

## ✨ Key Features

| Feature | Description |
|:--------|:------------|
| 🔥 **Safety Equipment Detection** | Detects fire extinguishers and other critical safety assets |
| 🗂️ **Clutter Classification** | Differentiates between dark clutter and light uncluttered environments |
| 📷 **Real-Time Webcam Inference** | Live object detection feed via webcam |
| 🖼️ **Image & Video Prediction** | Run inference on static images or video files |
| 🌐 **Flask Web Interface** | Upload images and view detections directly in the browser |
| 📊 **Evaluation Pipeline** | Precision, Recall, mAP@0.5, mAP@0.5:0.95, F1-score reporting |
| ⚡ **Baseline + Optimized Models** | Compare standard and fine-tuned model performance |
| 📈 **Confusion Matrix & CSV Output** | Detailed per-class performance breakdown |

---

## 📁 Project Structure

```text
HWB-ZENITHRA/
│
├── App/
│   └── yolo-web-app/              # Flask web application
│       ├── app.py                 # Main Flask entry point
│       ├── templates/
│       │   └── index.html         # Upload & result UI
│       ├── static/
│       │   ├── css/               # Stylesheets
│       │   └── uploads/           # Uploaded inference images
│       └── requirements.txt       # Web app dependencies
│
├── Model/
│   ├── dataset/                   # Dataset (images + labels in YOLO format)
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   ├── data.yaml                  # Dataset configuration file
│   ├── train_baseline.py          # Baseline model training script
│   ├── train_optimized.py         # Optimized model training script
│   ├── inference.py               # Image/video inference script
│   ├── webcam_inference.py        # Real-time webcam inference script
│   ├── evaluate.py                # Evaluation & metrics script
│   └── weights/
│       ├── baseline/
│       │   └── best.pt            # Baseline trained weights
│       └── optimized/
│           └── best.pt            # Optimized trained weights
│
├── Documentation/
│   ├── report.pdf                 # Project report
│   └── demo/                      # Demo screenshots & videos
│
├── requirements.txt               # Global project dependencies
└── README.md                      # This file
```

---

## 🛠️ Requirements

### Software

| Dependency | Version |
|:-----------|:--------|
| Python | ≥ 3.8 |
| PyTorch | ≥ 2.0 |
| Ultralytics (YOLOv8) | ≥ 8.0 |
| OpenCV | ≥ 4.7 |
| Flask | ≥ 2.3 |
| NumPy | ≥ 1.24 |
| Pandas | ≥ 2.0 |
| Matplotlib | ≥ 3.7 |
| Scikit-learn | ≥ 1.3 |

### Hardware

| Component | Minimum | Recommended |
|:----------|:--------|:------------|
| RAM | 8 GB | 16 GB |
| GPU VRAM | 4 GB (CUDA) | 8 GB+ |
| Storage | 5 GB free | 10 GB+ |

> ⚠️ **Note:** A CUDA-capable GPU is strongly recommended for training. CPU inference is supported but significantly slower.

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/HWB-ZENITHRA.git
cd HWB-ZENITHRA
```

### 2. Create & Activate a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify GPU (Optional but Recommended)

```python
import torch
print(torch.cuda.is_available())       # Should return True
print(torch.cuda.get_device_name(0))   # Your GPU name
```

---

## 🏋️ Training the Model

Ensure your dataset is organized under `Model/dataset/` and `data.yaml` is correctly configured.

### `data.yaml` Example

```yaml
path: ./dataset
train: images/train
val: images/val

nc: 3
names: ['fire_extinguisher', 'dark_clutter', 'light_unclutter']
```

### Train the Baseline Model

```bash
cd Model
python train_baseline.py
```

Trains `yolov8n.pt` (nano) with default hyperparameters for 50 epochs.

### Train the Optimized Model

```bash
python train_optimized.py
```

The optimized pipeline includes:

- Extended epochs (100+)
- Custom augmentation (mosaic, flipping, HSV shifts)
- Learning rate scheduling
- Early stopping with best checkpoint saving

> 📁 Trained weights are saved to `Model/weights/baseline/` and `Model/weights/optimized/` respectively.

---

## 🚀 Running Inference

### Image Inference

```bash
cd Model
python inference.py --source path/to/image.jpg --weights weights/optimized/best.pt
```

### Video Inference

```bash
python inference.py --source path/to/video.mp4 --weights weights/optimized/best.pt
```

### Real-Time Webcam Inference

```bash
python webcam_inference.py --weights weights/optimized/best.pt
```

> Press `Q` to quit the webcam stream.

### Common CLI Arguments

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--source` | required | Path to image, video, or `0` for webcam |
| `--weights` | required | Path to `.pt` model weights |
| `--conf` | `0.25` | Confidence threshold (0.0 – 1.0) |
| `--iou` | `0.45` | IoU threshold for NMS |
| `--save` | `False` | Save output to `runs/detect/` |
| `--show` | `True` | Display result in a window |

---

## 🌐 Web Application

The Flask web app allows users to upload images and receive annotated detection results directly in the browser.

### Setup

```bash
cd App/yolo-web-app
pip install -r requirements.txt
```

### Run the App

```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000`

### Features

- 📤 Upload `.jpg`, `.jpeg`, or `.png` images
- 🖼️ View annotated results inline
- 📋 See detected class labels and confidence scores
- 🔄 Reset and upload new images seamlessly

---

## 📊 Performance Metrics

After training, run the evaluation script to generate a full metrics report:

```bash
cd Model
python evaluate.py --weights weights/optimized/best.pt --data data.yaml
```

### Metrics Explained

| Metric | Description |
|:-------|:------------|
| **Precision** | Of all positive predictions, how many are actually correct |
| **Recall** | Of all actual positives, how many were correctly detected |
| **mAP@0.5** | Mean Average Precision at IoU threshold 0.50 |
| **mAP@0.5:0.95** | mAP averaged across IoU thresholds from 0.50 to 0.95 |
| **F1-Score** | Harmonic mean of Precision and Recall |

### Sample Results (Optimized Model)

| Class | Precision | Recall | mAP@0.5 | F1-Score |
|:------|:---------:|:------:|:-------:|:--------:|
| fire_extinguisher | 0.94 | 0.91 | 0.93 | 0.92 |
| dark_clutter | 0.88 | 0.85 | 0.87 | 0.86 |
| light_unclutter | 0.91 | 0.89 | 0.90 | 0.90 |
| **Overall** | **0.91** | **0.88** | **0.90** | **0.89** |

> ℹ️ Results may vary depending on dataset size, quality, and training configuration.

---

## 📂 Output Explanation

After running inference or training, the following outputs are generated:

| Output | Location | Description |
|:-------|:---------|:------------|
| Annotated Images | `runs/detect/predict/` | Original images with bounding boxes drawn |
| Annotated Videos | `runs/detect/predict/` | Frame-by-frame detected video output |
| `results.csv` | `runs/train/` | Per-epoch training metrics (loss, mAP, etc.) |
| `confusion_matrix.png` | `runs/train/` | Visual confusion matrix across all classes |
| `PR_curve.png` | `runs/train/` | Precision-Recall curve |
| `F1_curve.png` | `runs/train/` | F1 vs. confidence threshold curve |
| `weights/best.pt` | `runs/train/weights/` | Best model checkpoint |
| `weights/last.pt` | `runs/train/weights/` | Last epoch model checkpoint |

---

## 🔧 Advanced Usage

### Export Model to ONNX / TensorRT

```python
from ultralytics import YOLO

model = YOLO('weights/optimized/best.pt')

# Export to ONNX
model.export(format='onnx')

# Export to TensorRT (requires TensorRT installed)
model.export(format='engine', device=0)
```

### Multi-GPU Training

```python
# In train_optimized.py
model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    device='0,1'    # Use GPU 0 and GPU 1
)
```

### Adjust Confidence Threshold at Runtime

```bash
python inference.py --source image.jpg --weights weights/optimized/best.pt --conf 0.5
```

### Batch Inference on a Folder

```bash
python inference.py --source path/to/images_folder/ --weights weights/optimized/best.pt --save
```

---

## 🐞 Troubleshooting

### ❌ `CUDA out of memory`

**Cause:** Batch size too large for available VRAM.

**Fix:** Reduce batch size in the training script:

```python
model.train(..., batch=8)   # Reduce from 16 to 8
```

### ❌ `ModuleNotFoundError: No module named 'ultralytics'`

**Fix:**

```bash
pip install ultralytics
```

### ❌ Webcam not detected (`cv2.error`)

**Cause:** System camera index mismatch.

**Fix:** Try changing the camera index:

```bash
python webcam_inference.py --source 1   # Try 1 or 2 instead of 0
```

### ❌ Flask app shows no detections

**Cause:** Incorrect path to model weights in `app.py`.

**Fix:** Ensure the weights path is absolute or correctly relative to `app.py`.

### ❌ `data.yaml` not found during training

**Fix:** Provide an absolute path:

```python
model.train(data='/absolute/path/to/data.yaml', ...)
```

### ❌ Slow inference on CPU

**Recommendation:** Use a CUDA-capable GPU. Alternatively, export to ONNX for optimized CPU inference:

```bash
model.export(format='onnx')
```

---

## 📚 References & Citations

```bibtex
@software{yolov8_ultralytics,
  author  = {Glenn Jocher and Ayush Chaurasia and Jing Qiu},
  title   = {Ultralytics YOLOv8},
  year    = {2023},
  url     = {https://github.com/ultralytics/ultralytics},
  license = {AGPL-3.0}
}

@incollection{PyTorch,
  title     = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
  author    = {Paszke, Adam and others},
  booktitle = {Advances in Neural Information Processing Systems 32},
  year      = {2019},
  url       = {https://pytorch.org}
}

@article{opencv_library,
  author  = {Bradski, G.},
  title   = {The OpenCV Library},
  journal = {Dr. Dobb's Journal of Software Tools},
  year    = {2000},
  url     = {https://opencv.org}
}

@misc{lin2014microsoft,
  title   = {Microsoft COCO: Common Objects in Context},
  author  = {Tsung-Yi Lin and others},
  year    = {2014},
  url     = {https://cocodataset.org}
}

@misc{flask,
  title  = {Flask: A Lightweight WSGI Web Application Framework},
  author = {Pallets Projects},
  year   = {2010},
  url    = {https://flask.palletsprojects.com}
}
```

---

## 📄 License

This project is licensed under the **MIT License**.

```text
MIT License

Copyright (c) 2024 HWB-ZENITHRA Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

---

<p align="center">
  <strong>Built with ❤️ by the HWB-ZENITHRA Team</strong><br>
  <em>Making workplaces safer, one detection at a time.</em><br><br>
  ⭐ <strong>Star this repository</strong> if you find it useful!
</p>
