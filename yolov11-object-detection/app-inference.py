import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO
import torch
from collections import Counter

torch.classes.__path__ = []

# Load both YOLOv11 models (detection and segmentation)
model_det = YOLO("yolo11l.pt")          # Detection model
model_seg = YOLO("yolo11l-seg.pt")      # Segmentation model

# Object classes
classNames = ["person", " Bicycle", " CAR ", "Bike    ", "Sportscar", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "Bird", "bench", "Dog", "Horse",
    "dog", "horse", "sheep", "cow", "Zebra", "Giraffe", "Elephant", "Bear", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "ball", "Kite", "blue pen   ", "Snowboard", "mobile phone",
    "baseball glove", "skateboard", "bottle   ", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "phone ", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "book", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]


if "webcam_status" not in st.session_state:
    st.session_state.webcam_status = False

def detect_objects_y11(img, model, segmentation=False):
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    results = model(img, verbose=False)
    detected_objects = []

    for r in results:
        boxes = r.boxes
        masks = r.masks if segmentation and hasattr(r, 'masks') else None

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = round(float(box.conf[0]) * 100, 2)
            cls = int(box.cls[0])
            label = classNames[cls].strip()
            detected_objects.append(label)

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(img, f"{label} {confidence}%", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            if segmentation and masks is not None and i < len(masks.data):
                mask = masks.data[i].cpu().numpy()

                # Resize the mask to match the image dimensions
                mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]))

                colored_mask = np.zeros_like(img, dtype=np.uint8)
                rng = np.random.default_rng(cls)
                color = rng.integers(0, 255, size=(3,), dtype=np.uint8)
                for c in range(3):
                    colored_mask[:, :, c] = (mask_resized * color[c]).astype(np.uint8)
                img = cv2.addWeighted(img, 1.0, colored_mask, 0.5, 0)

    object_counts = Counter(detected_objects)

    y_offset = 30
    for obj, count in object_counts.items():
        cv2.putText(img, f"{obj}: {count}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        y_offset += 30

    return img, object_counts

def detect_realtime_objects(model, segmentation=False):
    st.session_state.webcam_status = True
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    cap.set(3, 640)
    cap.set(4, 480)

    emptyFrame = st.empty()
    object_count_placeholder = st.sidebar.empty()

    while st.session_state.webcam_status:
        success, img = cap.read()
        if not success:
            st.error("Error capturing video frame.")
            break

        detected_img, object_counts = detect_objects_y11(img, model, segmentation=segmentation)
        emptyFrame.image(cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB), use_container_width=True)

        object_count_html = "<div style='max-height: 300px; overflow-y: auto;'>"
        object_count_html += "<h4>Live Object Count</h4>"
        for obj, count in object_counts.items():
            object_count_html += f"<p>🔹 {obj}: {count}</p>"
        object_count_html += "</div>"

        object_count_placeholder.markdown(object_count_html, unsafe_allow_html=True)


    cap.release()
    cv2.destroyAllWindows()

def stop_webcam():
    st.session_state.webcam_status = False

def detect_objects_in_video(video_path, model, segmentation=False):
    cap = cv2.VideoCapture(video_path)
    emptyFrame = st.empty()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        detected_frame, object_counts = detect_objects_y11(frame, model, segmentation=segmentation)
        emptyFrame.image(detected_frame, channels="BGR", use_container_width=True)

        st.sidebar.subheader("Live Object Count")
        for obj, count in object_counts.items():
            st.sidebar.write(f"{obj}: {count}")

    cap.release()

# Streamlit UI
st.title("🚀 YOLOv11 Object Detection")
st.markdown("⚠️ **Note:** Please select only one option at a time — either use webcam detection or upload an image or an video. Running both simultaneously may cause conflicts.", unsafe_allow_html=True)
segmentation_enabled = st.checkbox("🧩 Enable Segmentation")

start = st.button("🎥 Start Webcam Detection")
stop = st.button("🛑 Stop Webcam")

# Image Upload
uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    with st.spinner("Processing image..."):
        image = Image.open(uploaded_file)
        image = np.array(image)
        model_used = model_seg if segmentation_enabled else model_det
        detected_image, object_counts = detect_objects_y11(image.copy(), model_used, segmentation=segmentation_enabled)
        
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, channels="RGB", caption="Original Image", use_container_width=True)
    with col2:
        st.subheader("Detected Objects")
        st.image(detected_image, channels="BGR", caption="Detected Image", use_container_width=True)
        st.subheader("Object Count")
        for obj, count in object_counts.items():
            st.write(f"{obj}: {count}")

# Video Upload
uploaded_video = st.file_uploader("📂 Upload a video", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_video.read())
    detect_objects_in_video("temp_video.mp4", model_seg if segmentation_enabled else model_det, segmentation=segmentation_enabled)

# Realtime Controls
if start:
    detect_realtime_objects(model_seg if segmentation_enabled else model_det, segmentation=segmentation_enabled)

if stop:
    stop_webcam()