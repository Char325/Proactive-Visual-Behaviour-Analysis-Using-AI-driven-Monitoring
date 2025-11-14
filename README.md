# 😴 Proactive Visual Behaviour Analysis Using AI-driven Monitoring: A Real-Time Driver Drowsiness Detection System

## 🌟 Overview

This project implements a low-cost, non-intrusive, and real-time driver drowsiness detection system. The core of the system is a deep learning model, **FastViT-DrowsyNet**, based on the Vision Transformer (ViT) architecture, which analyzes visual cues like eye closure and yawning to assess the driver's alertness.

The complete solution is an **integrated and end-to-end ecosystem** that combines:
1.  **On-device intelligence** using a Raspberry Pi 4 for real-time processing.
2.  An **Android mobile application** for instant alerts, real-time location display, and emergency notification.
3.  A **Fleet Management** dashboard via MQTT and a cloud platform for remote monitoring.

This approach addresses the silent but dangerous threat of driver fatigue on the road by providing a portable, scalable, and accurate solution.

---

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



*Figure 1.1: Proposed Model Architecture*

---

## 🛠️ Methodology & Deployment

The implementation is divided into three major interconnected components:

### 1. FastViT-DrowsyNet Model
The deep learning model is fine-tuned on a driver drowsiness dataset (DDD). The model leverages embeddings from the FastViT architecture to classify the driver's state.

### 2. Android Mobile Application
The mobile app, built using Kotlin DSL, provides several critical features:
* Providing instant alerts (Vibration + Sound) through the smartphone.
* Displaying the driver's real-time location using a map interface.
* Notifying an emergency contact in case of severe or repeated drowsiness detection.
* Presenting visual trends and logs of past drowsiness events.

### 3. Fleet Management
This component enables remote monitoring of vehicles and drivers:
* **GPS Data Acquisition:** Fetches GPS coordinates.
* **Data Publishing:** Uses MQTT to publish data (status and location) to the cloud (e.g., ThingSpeak).
* **Cloud Dashboard:** Data is visualized on a cloud dashboard, allowing a fleet manager to monitor vehicles.

---

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
