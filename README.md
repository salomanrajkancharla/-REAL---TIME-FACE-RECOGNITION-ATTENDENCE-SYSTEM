# 🎯 Real-Time Face Recognition & Attendance System

A computer vision-based application that automates attendance tracking by identifying individuals in real time through a live camera feed — eliminating the need for manual roll calls, registers, or biometric hardware.

## 📌 Overview

This system uses facial recognition to mark attendance automatically. It works in two phases:

1. **Enrollment** – Captures and encodes each person's facial data into the system's database.
2. **Recognition** – Continuously scans a live video feed, matches detected faces against stored encodings, and logs attendance with name, date, and time once a match is confirmed.

Ideal for schools, colleges, offices, and events looking to reduce proxy attendance and save administrative time.

## ✨ Features

- ✅ Real-time face detection and recognition
- ✅ Multi-face recognition in a single frame
- ✅ Automatic timestamping of attendance
- ✅ Attendance logs exported to CSV/database
- ✅ Simple enrollment process for new users
- ✅ Contactless and scalable solution

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** OpenCV, face_recognition / dlib, NumPy, Pandas
- **Database:** CSV / SQLite / MySQL / Firebase
- **Interface:** Tkinter / Streamlit / Flask
- **Hardware:** Webcam / CCTV Camera

## 📂 Project Structure

face-recognition-attendance/
├── dataset/ # Stored face images for enrollment
├── encodings/ # Saved face encodings
├── attendance/ # Attendance logs (CSV/DB)
├── src/
│ ├── enroll.py # Face enrollment script
│ ├── recognize.py # Real-time recognition & attendance marking
│ └── utils.py # Helper functions
├── requirements.txt
└── README.md


## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/face-recognition-attendance.git
cd face-recognition-attendance
pip install -r requirements.txt
```

## 🚀 Usage

**1. Enroll a new user**
```bash
python src/enroll.py --name "John Doe"
```

**2. Run real-time recognition & attendance**
```bash
python src/recognize.py
```

Attendance will be automatically logged in `attendance/attendance.csv` with name, date, and time.

## 🔮 Future Scope

- Liveness detection to prevent spoofing via photos
- Cloud-based database integration
- Web dashboard for attendance analytics
- Mobile app integration
- Multi-camera support for larger venues

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](../../issues).
