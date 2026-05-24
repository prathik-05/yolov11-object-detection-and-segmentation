# 🚀 Real-Time Object Detection & Segmentation using YOLOv11

An AI-powered real-time object detection and segmentation platform built using **YOLOv11**, **OpenCV**, and **Streamlit**. The application supports image, video, and webcam-based inference with dynamic switching between detection and segmentation models, live object counting, and interactive visual analytics.

---

# ✨ Features

- 🎯 Real-time Object Detection using `yolo11l.pt`
- 🧠 Real-time Object Segmentation using `yolo11l-seg.pt`
- 📸 Supports Image, Video, and Webcam Inputs
- 📊 Live Object Count Overlay
- 🔄 Dynamic Detection ↔ Segmentation Switching
- ⚡ Low-Latency Real-Time Inference
- 🖥️ Interactive Streamlit User Interface
- 📈 Class-wise Object Analytics
- 🎥 Webcam-Based Live Detection Pipeline

---

# 🛠️ Tech Stack

- Python
- YOLOv11 (Ultralytics)
- OpenCV
- Streamlit
- NumPy
- Pillow (PIL)

---

# 🏗️ Project Architecture

```text
Input Source
(Image / Video / Webcam)
            │
            ▼
     Streamlit Interface
            │
            ▼
 YOLOv11 Detection / Segmentation
            │
            ▼
   OpenCV Processing Pipeline
            │
            ▼
 Bounding Boxes / Masks / Analytics
            │
            ▼
      Real-Time Visualization
```

---

# 📂 Project Structure

```text
YOLOv11-Object-Detection/
│
├── app.py
├── requirements.txt
├── README.md
│
├── yolov11-models/
│   ├── yolo11l.pt
│   └── yolo11l-seg.pt
│
├── input_samples/
│
└── outputs/
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone Repository

```bash
git clone https://github.com/prathik-05/yolov11-object-detection.git

cd yolov11-object-detection
```

---

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3️⃣ Download YOLOv11 Models

Place the following model files inside:

```text
yolov11-models/
```

Required Models:

```text
yolo11l.pt
yolo11l-seg.pt
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

---

# 📸 Supported Inputs

- 🖼️ Images
- 🎥 Videos
- 📷 Webcam Stream

---

# 📊 Functionalities

- Real-time object detection
- Real-time segmentation masks
- Dynamic model switching
- Live object counting
- Class-wise visualization
- Webcam analytics
- Video frame processing

---

# 🔮 Future Improvements

- Object Tracking Integration
- FPS Optimization
- Multi-camera Support
- Export Detection Reports
- Cloud Deployment
- Advanced Analytics Dashboard

---

# 👨‍💻 Author

## Prathik Salla

- GitHub: https://github.com/prathik-05
- LinkedIn: https://www.linkedin.com/in/prathik-s07/

---

# 📜 License

This project is open-source and intended for educational, research, and learning purposes.
