# 🧠 Face Recognition Attendance System

A Python-based smart attendance system that uses real-time face recognition to automate attendance logging.
This project utilizes OpenCV and the `face_recognition` library to detect and identify known individuals, and logs their presence in a CSV file with timestamps.

---

## 📌 Features

- Real-time face detection using webcam
- Face recognition using `face_recognition` library
- Logs attendance with timestamps in CSV format
- Automatically avoids duplicate entries
- Easy-to-use and customizable

---

## 📁 Project Structure

Face-Recognition-Attendance/
├── faces/ # Folder containing images of known people
│ ├── person1.jpeg
│ ├── person2.jpeg
│ └── ...
├── main.py # Main script for running the attendance system
├── requirements.txt # Python dependencies
└── YYYY-MM-DD.csv # Automatically generated attendance log file

-Install Requirments
cmake
numpy
opencv-python
face_recognition
