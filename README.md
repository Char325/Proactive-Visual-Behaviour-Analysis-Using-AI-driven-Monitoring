# 😴 Proactive Visual Behaviour Analysis Using AI-driven Monitoring: A Real-Time Driver Drowsiness Detection System

## 🌟 Overview

This project implements a low-cost, non-intrusive, and real-time driver drowsiness detection system. The core of the system is a deep learning model, **FastViT-DrowsyNet**, based on the Vision Transformer (ViT) architecture, which analyzes visual cues like eye closure and yawning to assess the driver's alertness.

The complete solution is an **integrated and end-to-end ecosystem** that combines:
1.  **On-device intelligence** using a Raspberry Pi 4 for real-time processing.
2.  An **Android mobile application** for instant alerts, real-time location display, and emergency notification.
3.  A **Fleet Management** dashboard via MQTT and a cloud platform for remote monitoring.

This approach addresses the silent but dangerous threat of driver fatigue on the road by providing a portable, scalable, and accurate solution.

---
## ✨ Motivation
* **Safety Concern:** 
Road safety remains a major global concern, with driver drowsiness contributing to a significant number of traffic accidents every year. Drowsy driving is a leading cause of fatal crashes, especially during long highway journeys, nighttime driving, or monotonous routes.

* **The Silent Threat:** 
Unlike drunk or distracted driving, fatigue often goes unnoticed until it is too late, making it a silent but dangerous threat on the road.

* **Limitations of Current Systems:** 
Most existing solutions—such as steering wheel movement sensors or lane deviation trackers—either require specialized hardware or are only effective in specific scenarios. Additionally, many commercial driver monitoring systems are expensive and inaccessible to everyday users or small-scale transport operators.
---

## 🛠️ Built With
* <img src='https://img.shields.io/badge/Raspberry%20Pi-A22846?style=for-the-badge&logo=Raspberry%20Pi&logoColor=white' />
* <img src='https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white' />
* <img src='https://img.shields.io/badge/Kotlin-B125EA?style=for-the-badge&logo=kotlin&logoColor=white' />
* <img src='https://img.shields.io/badge/OpenCV-27338e?style=for-the-badge&logo=OpenCV&logoColor=white' />

## 💡 Core Concept

The system proactively monitors signs of fatigue using a deep learning-based Vision Transformer (ViT) model.

* **Input Data:** The ViT model processes facial landmarks, specifically the **Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)**.
    * **EAR** helps identify frequent or prolonged eye closure, a hallmark of microsleep episodes.
    * **MAR** detects yawning patterns, another strong indicator of fatigue.
* **Hardware:** The detection system is deployed on a **Raspberry Pi 4 (RPi)**, with a camera module continuously capturing live video of the driver's face.
* **Alerting:** Upon detection, the system triggers a **local auditory alert** to immediately regain the driver's attention.
* **Location:** It fetches the driver's real-time **GPS coordinates** using the Neo 6M GPS module.
* **Communication:** Drowsiness status and GPS data are sent to a connected **Android mobile application** via a Flask-based REST API, and to a cloud via **MQTT** for fleet management.

#### Proposed Model Architecture



<img width="760" height="533" alt="image" src="https://github.com/user-attachments/assets/2b44dcdc-42df-4bf7-a18a-9fb09bef393a" />


---

## 👁️👁️ Dataset & Preprocessing Steps
The Kaggle Driver Drowsiness Dataset (DDD) serves as the foundational data source for
training and evaluating our drowsiness detection model. This dataset is specifically curated
with labeled images of drivers’ ears, where each image is associated with a correspond-
ing drowsiness level (typically categorized as ’drowsy’ or ’Not Drowsy’). To optimize the
model’s learning and generalization capabilities, a series of crucial preprocessing steps such as image resizing, pixel value normalization etc. are
applied to each image

** Some drowsy image examples from the DDD Dataset **
## 🛠️ Methodology & Deployment

The implementation is divided into three major interconnected components:

## 🧠 FastViT-DrowsyNet Model: Deep Dive

The **FastViT-DrowsyNet** model is the neural core of the system, specifically architected to provide high accuracy while maintaining the low computational overhead required for real-time deployment on embedded systems like the Raspberry Pi.

### 1. Hybrid Architecture: Efficiency First

The model is built upon the **FastViT** (Fast Hybrid Vision Transformer) architecture. This foundational choice ensures an optimal balance between deep feature extraction and inference speed.

| Feature | Design Rationale | Benefit for Edge Deployment |
| :--- | :--- | :--- |
| **Hybrid Structure** | Combines the local feature extraction of Convolutional Neural Networks (CNNs) with the global context modeling of Vision Transformers (ViT). | Higher accuracy than pure CNNs, but much faster than standard ViT models, making it suitable for the Raspberry Pi 4. |

| **Large Kernel Convolutions** | Integrated into the Feed Forward Network (FFN) layers. | Expands the model's effective receptive field to capture large-scale drowsiness cues (like full yawns) without relying on expensive self-attention across the entire image. |

<img width="786" height="1398" alt="image" src="https://github.com/user-attachments/assets/c2347952-9263-4eec-8dd5-2636c93cf772" />
Raspberry with PiCam5 Setup

### 2. Multi-Modal Feature Fusion

The model uses a crucial multi-modal input strategy, combining two types of features to enhance robustness and interpretability:

| Feature Type | Specific Cues | Calculation/Source |
| :--- | :--- | :--- |
| **Geometric Cues** (Handcrafted) | **Eye Aspect Ratio (EAR)** | Calculated from facial landmarks (using a library like dlib) to quantify eye closure. |
| **Geometric Cues** (Handcrafted) | **Mouth Aspect Ratio (MAR)** | Calculated from facial landmarks to quantify mouth opening (yawning). |
| **Deep Features** (Learned) | **FastViT Embeddings** | Features extracted by the FastViT backbone from the cropped face image, capturing subtle texture and high-level patterns. |

This hybrid feature fusion is critical: it allows the model to learn the spatial context of drowsiness (via ViT embeddings) while simultaneously using the universally proven geometric cues (EAR and MAR) that define drowsiness behaviors (prolonged eye closure or yawning).

### 3. Real-Time Classification

The final stage of the model performs the binary classification:
* **Input:** The fused feature vector (containing both ViT embeddings and EAR/MAR data) is fed into the final fully-connected layers.
* **Loss Function:** Training is performed using **BCEWithLogitsLoss** (Binary Cross Entropy with logits loss).
* **Output:** The model predicts the driver's state as one of two classes: **'Drowsy'** or **'Not Drowsy'**.
* **Temporal Smoothing:** To reduce the effect of transient noise and natural blinks, the geometric features (EAR/MAR) are typically smoothed across sequential video frames, ensuring the model classifies true behavioral *patterns* rather than isolated, instantaneous changes.

### 2. Android Mobile Application
The mobile app, built using Kotlin DSL, provides several critical features:
* Providing instant alerts (Vibration + Sound) through the smartphone.
* Displaying the driver's real-time location using a map interface.
* Notifying an emergency contact in case of severe or repeated drowsiness detection.
* Presenting visual trends and logs of past drowsiness events.
  <img width="399" height="816" alt="image" src="https://github.com/user-attachments/assets/e36a0d14-9544-4749-b2c5-1e2eb8b721e0" />


### 3. Fleet Management
This component enables remote monitoring of vehicles and drivers:
* **GPS Data Acquisition:** Fetches GPS coordinates.
* **Data Publishing:** Uses MQTT to publish data (status and location) to the cloud (e.g., ThingSpeak).
* **Cloud Dashboard:** Data is visualized on a cloud dashboard, allowing a fleet manager to monitor vehicles.
<img width="1936" height="757" alt="image" src="https://github.com/user-attachments/assets/84e84d34-744d-42f0-8aa2-689bc57e7ce5" />


---
# ⚙️ FastViT-DrowsyNet: Code Setup & Usage Guide

This guide details the steps required to set up the environment, download the necessary data, and execute the training and prediction scripts for the **FastViT-DrowsyNet** model.

## 📦 Prerequisites

Ensure you have the following software and libraries installed:

1.  **Python 3.x**
2.  **Required Libraries:** (Install via `pip install -r requirements.txt`)
    * Torch/Torchvision
    * OpenCV (`opencv-python`)
    * dlib (Requires system-level build tools/CMake)
    * Pandas, NumPy, etc.
3.  **Jupyter Notebook** (or Jupyter Lab) for running the `.ipynb` files.
4.  **Kaggle Account** (for downloading the DDD dataset).

---

## 💾 Step 1: Data Setup and Preprocessing

The model uses the **Kaggle Driver Drowsiness Dataset (DDD)**.

### 1.1 Download the Dataset

1.  Obtain the **DDD.zip** file from Kaggle (or upload the file if it's already secured).
2.  Place the **DDD.zip** file into the root of this project directory.
3.  Unzip the file. The dataset is expected to contain organized images of driver faces.

### 1.2 Prepare Annotations

The dataset utilizes **CSV files** to map image paths to calculated features like **EAR** (Eye Aspect Ratio).

* `annotations.csv`
* `train_annotations.csv`
* `val_annotations.csv`
* `test_annotations.csv`

The script `dataset.py` is responsible for defining how the model loads data based on these annotation files, linking the image path to its corresponding classification label and EAR value.

---

## 🔬 Step 2: Feature Calculation Scripts

The model relies on geometric features (EAR and MAR) calculated from facial landmarks.

### 2.1 Install dlib Dependencies

Ensure you have the necessary dlib dependency for facial landmark detection:
* `shape_predictor_68_face_landmarks.dat`: This file must be present in the root directory to run the landmark detection routines.

### 2.2 Calculate EAR and MAR

The core logic for calculating the geometric features is likely housed in `ddd.py`.

* **`ddd.py`:** This script calculates the **Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)** from the detected facial landmarks. These values are often pre-calculated and stored in the CSV files.

---

## 🤖 Step 3: Model Training and Deployment

### 3.1 Training the Model

1.  **Model Definition:** The `fastvit.ipynb` notebook likely contains the initial model setup and potentially some training/fine-tuning steps. The base model weights (`fastvit_t8.pth.tar`) are likely used for transfer learning.
2.  **Training Script:** Execute the main training script.
    ```bash
    python train_fastvit.py
    ```
    * This script reads data paths and EAR values from the CSV files (`train_annotations.csv`, `val_annotations.csv`) via `dataset.py`.
    * It fine-tunes the **FastViT** model (referred to as **FastViT-DrowsyNet** in the report) on the DDD.
    * The best performing model weights will be saved as **`best_loss.pth`**.

### 3.2 Prediction and Real-Time Use

1.  **Prediction Script:** Use `predict.py` to test the trained model on test data or use it for inference.
    ```bash
    python predict.py
    ```
2.  **Real-Time Deployment:** The file structure suggests that the `dds.py` script is the final **Driver Drowsiness System** (DDS) script. This is likely the real-time script deployed on the Raspberry Pi:
    * It loads the saved weights (`best_loss.pth`).
    * It uses **OpenCV** to capture live video.
    * It uses **dlib** for real-time facial landmark detection.
    * It calculates the EAR/MAR.
    * It feeds the features into the loaded **FastViT-DrowsyNet** model for classification.

---

## 📁 Repository File Manifest

| File/Folder | Purpose | Core Component |
| :--- | :--- | :--- |
| `DDD.zip` | The primary dataset archive (Kaggle Driver Drowsiness Dataset). | Data |
| `fastvit_t8.pth.tar` | Pre-trained weights for the base FastViT model (used for transfer learning). | Model |
| `best_loss.pth` | The final trained model weights after fine-tuning. | Model |
| `train_fastvit.py` | Main script for training and validation. | Training |
| `dataset.py` | Defines the PyTorch Dataset class, handling image loading and annotation parsing. | Data Loader |
| `train_annotations.csv`, `val_annotations.csv`, etc. | Annotation files mapping images to labels/EAR values. | Data |
| `ddd.py` | Utility script, likely containing the logic for EAR/MAR calculation. | Feature Engineering |
| `shape_predictor_68_face_landmarks.dat` | Necessary dlib file for detecting 68 facial landmarks. | dlib Dependency |
| `predict.py` | Script for testing inference on new images or videos. | Inference |
| `dds.py` | Final real-time driver drowsiness system script (likely for RPi deployment). | Deployment |

## 📋 Key Technologies and Abbreviations

| Abbreviation | Full Form | Description |
| :--- | :--- | :--- |
| **FastViT** | Fast hybrid Vision Transformer | Developed by Apple, specifically for image classification. |
| **EAR** | Eye Aspect Ratio | Metric used to quantify eye closure. |
| **MAR** | Mouth Aspect Ratio | Metric used to detect yawning patterns. |
| **RPi** | Raspberry Pi | Lightweight embedded computing platform for deployment. |
| **dlib** | Face Recognition Library | Used for locating facial landmarks. |
| **MQTT** | Message Queuing Telemetry Transport | Lightweight messaging protocol used for sending data to the cloud. |
| **BCEWithLogitsLoss** | Binary Cross Entropy with logits loss | Loss function used during the training process. |
| **API** | Application Programming Interface | Flask-based REST API is used for communication between RPi and the mobile app. |
