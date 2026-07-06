# Driver Drowsiness Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv)
![License](https://img.shields.io/github/license/MinhalHusain/Drowsiness_Detection_System)
![Last Commit](https://img.shields.io/github/last-commit/MinhalHusain/Drowsiness_Detection_System)


# 🚗 Real-Time Driver Drowsiness Detection System

## 📌 Problem Statement

Driver fatigue and drowsiness are among the leading causes of road accidents, especially during **night-time long-haul driving** such as truck and highway transport.
Traditional safety measures fail because they **do not monitor the driver’s physical alertness in real time**.

There is a strong need for a **real-time, automated, and intelligent system** that can:

* Continuously monitor a driver
* Detect whether the driver is **awake or drowsy**
* Trigger alerts before an accident occurs
* Be extendable to **IoT-based safety systems**

---

## 💡 Our Solution

This project implements a **real-time drowsiness detection system** using **Deep Learning and Computer Vision**.
[**Working Video**](https://drive.google.com/file/d/17plu9QvZDoBmOS-e43iu3kOzNvllW2Fn/view?usp=sharing)
### Key Idea

* Train a **binary image classification model** (`Awake` vs `Drowsy`)
* Use a **pretrained ResNet-18** for robust feature extraction
* Perform **live face detection via webcam**
* Predict driver state in **real time**
* Raise alerts if drowsiness persists

### Why this works

* CNNs capture facial fatigue patterns
* Transfer learning improves accuracy with limited data
* Temporal frame tracking reduces false positives

---

## 🧠 System Architecture

**Pipeline Flow:**

```
Webcam Frame
     ↓
Face Detection (OpenCV Haar Cascade)
     ↓
Face Cropping & Preprocessing
     ↓
ResNet-18 Classifier
     ↓
Awake / Drowsy Prediction
     ↓
Alert Trigger (if drowsy persists)
```

---

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning:** PyTorch, Torchvision
* **Model:** ResNet-18 (Transfer Learning)
* **Computer Vision:** OpenCV
* **Image Processing:** Pillow, NumPy
* **Dataset Handling:** YOLO-style labels
* **Deployment Ready:** IoT & Edge-compatible

---

## 📂 Project Structure

```
├── Data PreProcessing/
├── Dataset/
├── Model/
├── Model Training/
└── README.md
```

---

## ⚙️ How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Dataset Preparation

Ensure the dataset follows this structure:

```
images/
 ├── img1.jpg
 ├── img2.jpg

labels/
 ├── img1.txt
 ├── img2.txt
```

**Label format (YOLO-style):**

```
0 → Awake
1 → Drowsy
```

---

### 3️⃣ (Optional) Preprocess Data

```bash
python datapreprocessing.py
```

This step:

* Splits dataset into train/test
* Saves metadata and NumPy arrays (optional)

---

### 4️⃣ Train the Model

```bash
python ResNetmodel.py
```

This will:

* Train ResNet-18 on your dataset
* Save the trained model as:

```
drowsiness_detection_model.pth
```

---

### 5️⃣ Run Real-Time Detection

```bash
python runmodel.py
```

Controls:

* Webcam opens automatically
* Press **`q`** to exit

---

## 🚨 Output Behavior

* **Green Box:** AWAKE
* **Red Box:** DROWSY
* **Alert Trigger:**
  If drowsiness continues for multiple frames, a **DROWSINESS ALERT** is displayed.

---

## 🌐 Real-World Applications

* 🚛 **Truck & Bus Driver Monitoring**
* 🚗 Night-time highway safety
* 🏭 Industrial machine operator alertness
* 🧠 Workplace fatigue monitoring
* 🔌 **IoT integration** with:

  * Buzzers
  * Seat vibration
  * Smart dashboards
  * Fleet monitoring systems

---

## 🔮 Future Enhancements

* Eye Aspect Ratio (EAR) + blink analysis
* Temporal models (LSTM / sliding window)
* Face detection upgrade (MTCNN / RetinaFace)
* Edge deployment (Jetson / Raspberry Pi)
* IoT alerts (GSM / MQTT / CAN bus)
* Driver fatigue analytics dashboard

---

## Contributors

* Minhal Husain
* Garvit Audichya

This project was developed collaboratively as part of an academic project.

---