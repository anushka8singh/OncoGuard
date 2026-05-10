# OncoGuard – AI-Powered Medical Imaging Analysis System

> **Project Status:** Ongoing Development (Work in Progress)

## Overview

OncoGuard is an AI-powered medical imaging analysis system designed to assist in the detection and classification of tumors from medical scans using Deep Learning and Computer Vision.

The project uses transfer learning with ResNet50, GradCAM explainability, FastAPI backend services, and a scalable deployment pipeline using Docker and Kubernetes.

The system currently supports:

* Brain Tumor Classification
* Breast Tumor Classification
* Multi-class prediction
* Explainable AI using GradCAM heatmaps
* REST API-based prediction system
* Containerized deployment architecture

This project was built as an end-to-end AI engineering project covering:

* Machine Learning
* Deep Learning
* Computer Vision
* Backend Development
* API Design
* Docker
* Kubernetes
* MLOps concepts

---

# Features

## Brain Tumor Classification

Classifies MRI scans into:

* Glioma
* Meningioma
* Pituitary Tumor
* Normal

## Breast Tumor Classification

Classifies ultrasound scans into:

* Benign
* Malignant
* Normal

## Explainable AI (GradCAM)

Generates heatmaps highlighting regions of the image that influenced the model’s prediction.

## REST API

Provides prediction endpoints using FastAPI.

## Scalable Architecture

Designed for Docker and Kubernetes deployment.

---

# Tech Stack

| Category                | Technology         |
| ----------------------- | ------------------ |
| Programming Language    | Python             |
| Deep Learning Framework | PyTorch            |
| Computer Vision         | OpenCV             |
| Model Architecture      | ResNet50           |
| Explainable AI          | GradCAM            |
| API Framework           | FastAPI            |
| Frontend                | React.js (Planned) |
| Containerization        | Docker             |
| Orchestration           | Kubernetes         |
| Version Control         | Git + GitHub       |

---

# Project Architecture

```text
Medical Image
      ↓
Image Preprocessing
      ↓
ResNet50 Deep Learning Model
      ↓
Tumor Classification
      ↓
GradCAM Heatmap Generation
      ↓
FastAPI Backend Response
      ↓
Frontend Visualization
```

---

# Dataset Structure

## Brain Dataset

```text
data/brain/
    train/
        glioma/
        meningioma/
        normal/
        pituitary/

    test/
        glioma/
        meningioma/
        normal/
        pituitary/
```

## Breast Dataset

```text
data/breast/
    train/
        benign/
        malignant/
        normal/

    test/
        benign/
        malignant/
        normal/
```

---

# Model Training Pipeline

The project uses transfer learning with a pretrained ResNet50 model.

## Training Workflow

```text
Dataset
   ↓
Image Preprocessing
   ↓
Transfer Learning (ResNet50)
   ↓
Feature Extraction
   ↓
Fine-Tuning
   ↓
Model Saving
```

---

# Transfer Learning Strategy

The model training was performed in two stages:

## Stage 1 – Feature Extraction

* All pretrained layers frozen
* Only final classification layer trained
* Faster convergence
* Lower memory usage

## Stage 2 – Fine-Tuning

* Unfroze deeper layers (`layer4`)
* Fine-tuned model on medical images
* Improved localization and classification performance

---

# Image Preprocessing

The following preprocessing pipeline is applied before training and inference:

* Image resizing to 224×224
* Tensor conversion
* Image normalization
* Random horizontal flipping (augmentation)

Normalization values:

```python
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
```

---

# GradCAM Explainability

GradCAM is used to generate heatmaps that visualize the regions influencing model predictions.

This improves:

* Model interpretability
* Medical transparency
* Trustworthiness of predictions

Example pipeline:

```text
Input Image
     ↓
Model Prediction
     ↓
Gradient Extraction
     ↓
Activation Map Generation
     ↓
Heatmap Overlay
```

---

# Folder Structure

```text
OncoGuard/
│
├── data/
│   ├── brain/
│   └── breast/
│
├── models/
│   ├── brain_model.pth
│   └── breast_model.pth
│
├── src/
│   ├── train_brain.py
│   ├── train_breast.py
│   ├── predict.py
│   ├── gradcam.py
│   └── preprocess.py
│
├── api/
├── frontend/
├── docker/
├── k8s/
├── requirements.txt
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/your-username/OncoGuard.git
cd OncoGuard
```

---

# Create Virtual Environment

## Windows

```bash
python -m venv venv
venv\Scripts\activate
```

## Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Train Models

## Brain Tumor Model

```bash
cd src
python train_brain.py
```

## Breast Tumor Model

```bash
python train_breast.py
```

---

# Model Outputs

After training:

```text
models/
    brain_model.pth
    breast_model.pth
```

---

# Run Predictions

```bash
python predict.py
```

---

# FastAPI Backend (Planned / In Progress)

The backend service is being developed using FastAPI.

Planned API endpoint:

```http
POST /predict
```

Expected workflow:

```text
Upload Image
      ↓
Preprocess Image
      ↓
Run Model Prediction
      ↓
Generate GradCAM Heatmap
      ↓
Return JSON Response
```

Example response:

```json
{
  "prediction": "glioma",
  "confidence": 0.94,
  "heatmap": "heatmap.jpg"
}
```

---

# Docker Support (Planned)

The project is being containerized using Docker for reproducible deployment.

Planned services:

* FastAPI backend
* React frontend
* MongoDB

---

# Kubernetes Deployment (Planned)

The project is designed to support Kubernetes deployment for scalability.

Planned Kubernetes resources:

* Deployment
* Service
* Ingress
* Horizontal Pod Autoscaler

---

# Challenges Faced

During development, several engineering challenges were addressed:

* Handling large medical image datasets on limited hardware
* Preventing memory crashes during training
* Implementing transfer learning efficiently
* Improving GradCAM localization quality
* Structuring multi-class datasets correctly
* Managing Windows multiprocessing issues in PyTorch DataLoader

---

# Key Learnings

This project helped in understanding:

* Deep Learning fundamentals
* CNN architectures
* Transfer Learning
* Image preprocessing pipelines
* Explainable AI techniques
* API development with FastAPI
* MLOps and deployment workflows
* Scalable AI system design

---

# Future Improvements

Planned future enhancements include:

* React frontend dashboard
* Authentication system
* MongoDB integration
* Cloud deployment
* CI/CD pipeline
* Model monitoring
* Improved segmentation models
* Attention-based architectures

---

# Git Workflow

The project follows incremental Git commits for each development phase:

* Dataset preprocessing
* Model training
* GradCAM integration
* Backend API development
* Dockerization
* Kubernetes deployment

---

# License

This project is intended for educational and research purposes.

