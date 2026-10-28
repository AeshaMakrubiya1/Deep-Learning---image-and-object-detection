## 🧠 CV_PR3 — Image and Object detection

---

**From classical image processing to deep-learning object detection — a real-time pipeline built
across morphology, bitwise logic, histograms, face detection (YuNet), and object detection (YOLOv8),
integrated into a single annotated video stream.**

---

## 📌 Topics Covered

| # | Topic | Description |
|---|-------|-------------|
| 1 | **Morphological Operations** | Erosion, dilation, opening, closing — plus kernel shape/size comparison for noise removal |
| 2 | **Bitwise Operations & Histograms** | AND/OR/XOR/NOT masking, ROI overlays, grayscale & colour histograms, brightness/contrast tuning |
| 3 | **Face Detection (YuNet)** | OpenCV 5.0's CNN-based `FaceDetectorYN` — static images, real-time webcam, threshold tuning, privacy blur |
| 4 | **Object Detection (YOLOv8)** | Ultralytics YOLOv8-Nano on COCO — static + real-time detection, confidence/IoU tuning, per-class summary |
| 5 | **Integrated Pipeline** | Combined face + object detection on one stream, FPS benchmarking, final results comparison |

---

## 🗂️ Dataset & Model Sources

| Resource | Source | Notes |
|---|---|---|
| Static image pack | [OpenCV sample data](https://github.com/opencv/opencv/tree/master/samples/data) (`lena.jpg`, `baboon.jpg`, `fruits.jpg`, `home.jpg`, `messi5.jpg`, `vehicle-traffic.jpg`) + personal phone photos | Morphology, bitwise masking, histograms, detection |
| Face detector weights | [`face_detection_yunet_2023mar.onnx`](https://github.com/opencv/opencv_zoo) — OpenCV Zoo | Loaded via `cv2.FaceDetectorYN_create()` |
| Object detector weights | `yolov8n.pt` — Ultralytics YOLOv8-Nano, pretrained on 80-class COCO | Auto-downloads via `from ultralytics import YOLO; YOLO('yolov8n.pt')` |
| Live input | Laptop webcam (`cv2.VideoCapture(0)`) | Used for real-time face, YOLO, and integrated pipeline tasks |

**Licenses:** OpenCV sample data — BSD-3-Clause · Ultralytics YOLOv8 — AGPL-3.0 (pretrained weights free for academic/educational use)

---

## ⚙️ Setup & Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**
```
opencv-python==5.0.0.*
opencv-contrib-python==5.0.0.*
ultralytics
numpy
matplotlib
pandas
```

Keep all static images in `/data/images` and downloaded model weights (including
`face_detection_yunet_2023mar.onnx`) in `/data/models` so paths stay consistent across the notebook.

Then open and run:

```bash
jupyter notebook CV_PR3.ipynb
```

---

## 📁 Repository Structure

```
├── CV_PR3.ipynb                       # Main notebook (all 5 tasks)
├── CV_PR3.html                        # Exported HTML (Restart & Run All)
├── requirements.txt
├── data/
│   ├── images/                        # baboon.jpg, fruits.jpg, home.jpg, lena.jpg, messi5.jpg, vehicle-traffic.jpg, ...
│   └── models/
│       ├── face_detection_yunet_2023mar.onnx
│       └── yolov8n.pt
├── plots/
│   ├── morphology_grid.png
│   ├── histogram_comparison.png
│   ├── face_detection_comparison.png
│   ├── yolo_detections.png
│   ├── fps_comparison.png
│   └── results_table.png
└── README.md
```

---


## 🖼️ Output & Results

### Task 1 — Morphological Operations
Environment confirmed on **OpenCV 5.0.0**. Erosion (local-minimum filter), dilation (local-maximum filter),
opening (erosion → dilation, removes small noise while preserving shape), and closing (dilation → erosion,
fills small holes) were applied on a binarized test image, followed by a kernel-shape × kernel-size (Rect /
Ellipse / Cross × 3×3 / 5×5 / 9×9) comparison grid.


### Task 2 — Bitwise Operations & Histograms
AND/OR/XOR/NOT demonstrated between circle and rectangle masks, with an applied masked-overlay use-case on a
real photo. Grayscale and overlaid B/G/R histograms were plotted, and `convertScaleAbs` was used to compare
brightened (`alpha=1.5, beta=30`) vs. darkened (`alpha=0.7, beta=-30`) versions against the original.

![Histogram comparison](plots/histogram_comparison.png)

### Task 3 — Face Detection (YuNet)
`cv2.FaceDetectorYN` detected faces with bounding boxes, 5-point landmarks, and confidence scores across
static images and a live webcam stream. Score-threshold sweep (0.5 / 0.7 / 0.9) showed the trade-off between
missed detections and false positives, and a Gaussian-blur privacy/anonymisation filter was applied to each
detected face region.

### Task 4 — Object Detection (YOLOv8)
`yolov8n.pt` detected COCO-class objects (person, car, dog, chair, etc.) on static images and webcam frames.
Confidence (0.25 / 0.5 / 0.75) and IoU (0.3 / 0.5 / 0.7) thresholds were tuned, including a duplicate-box
example resolved by Non-Max Suppression at higher IoU. A per-class detection bar chart summarised counts
across all test media.


### Task 5 — Integrated Pipeline
Face (green boxes) and object (red boxes) detection layers were combined on a single webcam stream.
Morphological opening as a pre-cleaning step reduced YOLO detections on a noisy frame from **7 → 4** objects
(removing spurious noise-driven boxes). FPS was benchmarked over a 100-frame sample for each configuration:

#### Final Comparison Table

![Results table](plots/results_table.png)

---

## 💡 Reflection

Classical pre-processing (morphology, bitwise/histogram analysis) is lightweight and extremely fast, making it
ideal for cleaning noisy camera frames cheaply before detection. YuNet delivers fast, accurate face detection
with low compute cost, while YOLOv8-Nano detects 80 object classes in real time with a good accuracy-to-speed
trade-off. The best combination for real-time monitoring was **morphological opening → YuNet → YOLOv8n**,
which reduced noise while keeping the pipeline real-time.

- **Edge deployment (e.g. Raspberry Pi / Jetson Nano):** keep `yolov8n.pt`, raise the YuNet score threshold
  (e.g. 0.7 → 0.8) to cut false positives, process at lower resolution (640×480), and apply only light
  pre-processing.
- **Cloud server deployment:** upgrade to a larger YOLO variant (`yolov8s.pt` or bigger) for higher accuracy,
  use higher-resolution frames, and slightly lower the YuNet threshold to catch more faces.


## 🛠️ Tools & Frameworks

- **OpenCV (cv2) 5.0.0** — classical image processing, YuNet face detection
- **Ultralytics YOLOv8** — deep-learning object detection
- **NumPy** — array/image manipulation
- **Matplotlib** — visualization and plotting
- **Pandas** — tabular results
- **Jupyter Notebook** — development environment (Python 3.x)


---

