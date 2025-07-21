
# 🚀 Real-Time Object Detection using YOLOv11

This is a real-time object detection and segmentation web application built using **YOLOv11**, **OpenCV**, and **Streamlit**. The app supports detection from images, videos, and webcam input, with optional segmentation and object count overlays.

---

## 🔧 Features

- 🧠 Object Detection (`yolo11l.pt`) and Segmentation (`yolo11l-seg.pt`)
- 📸 Supports image, video, and webcam inputs
- 📊 Live object count overlay on screen
- 🖼️ Sidebar toggle for segmentation
- ⚡ Real-time processing with Streamlit interface

---

## 🛠 Technologies Used

- Python
- YOLOv11 (Ultralytics)
- OpenCV
- Streamlit
- NumPy
- PIL (Pillow)

---

## 📁 Project Structure

YOLOv11_Object_Detection/
  |
  ├── app.py # Main Streamlit application
  ├── requirements.txt # Python dependencies
  ├── README.md # Project documentation
  ├── yolov11-models/ # YOLOv11 .pt model files
  ├── input_samples/ # Test images or videos


---

## ⚙️ How to Run

Clone the repository:

```bash
git clone https://github.com/prathik-05/yolov11-object-detection
cd yolov11-object-detection

Install required libraries:

pip install -r requirements.txt

Download and place the following models into the yolov11-models

   --> yolo11l.pt

   --> yolo11l-seg.pt

Then run the app:

streamlit run app-interface.py


👤 Author

S. Prathik Reddy
B.Tech CSE (Data Science), Ace Engineering College
GitHub: prathik-05


📜 License

This project is open-source and free to use for educational and academic purposes.

---

✅ Copy this and save it as `README.md` in your project folder.

Once you upload this to GitHub, your project will look **polished and professional**. Let me know when it’s live so I can move on to update your **resume** and help with **LinkedIn setup