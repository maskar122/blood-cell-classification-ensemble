## Blood Cell Classification - Ensemble Model

##  📋 Project Overview
This project implements an ensemble deep learning system for classifying blood cell images into four types: Eosinophil, Lymphocyte, Monocyte, and Neutrophil. The system combines three state-of-the-art CNN architectures (VGG16, ResNet50, InceptionV3) with a smart ensemble mechanism that selects the most confident prediction.

https://img.shields.io/badge/Blood-Cell%2520Classification-red
https://img.shields.io/badge/Deep-Learning-blue
https://img.shields.io/badge/TensorFlow-2.12%252B-orange

## 🖼️ Project Demo

![Blood Cell Classification - Ensemble Model](https://github.com/maskar122/blood-cell-classification-ensemble/blob/55f70d9d55ae5a1599ca0eabbebba68b3f037427/images/Screenshot%20(598).png)
![Blood Cell Classification - Ensemble Model](https://github.com/maskar122/blood-cell-classification-ensemble/blob/55f70d9d55ae5a1599ca0eabbebba68b3f037427/images/Screenshot%20(599).png)


## 🏆 Performance Highlights
Model	Test Accuracy	Precision	Recall	F1-Score
InceptionV3	100%	1.00	1.00	1.00
ResNet50	97.69%	0.98	0.98	0.98
VGG16	99.10%	0.99	0.99	0.99
Ensemble	Smart Selection	Optimal	Optimal	Optimal


## 🎯 Key Features
Multi-Model Ensemble: Combines VGG16, ResNet50, and InceptionV3 predictions

Smart Confidence Selection: Automatically selects the most confident model

Web Interface: Streamlit-based user-friendly application

Mobile Optimization: TensorFlow Lite models for efficient deployment

Data Augmentation: Advanced preprocessing for robust training

## 📁 Project Structure

blood-cell-classification/
│
├── models/
│   ├── VGG16_model.tflite

│   ├── ResNet_model.tflite

│   ├── inception_model.tflite

│   └── training_notebooks/
│       ├── vgg16.ipynb
│       ├── resnet.ipynb
│       └── inception_v3.ipynb
│
├── app/
│   └── app.py              # Streamlit web application
│

├── ensemble/
│   └── ensemble.ipynb      # Ensemble model implementation
│

├── data/
│   └── dataset2-master/    # Blood cell images dataset
│

└── requirements.txt

## 🚀 Quick Start
Prerequisites
Python 3.8+

TensorFlow 2.12+

Streamlit

##  🔬 Technical Implementation
Model Architectures
1. VGG16
Base Model: Pre-trained on ImageNet

Custom Layers: GlobalAveragePooling2D, Dense(256), Dropout(0.3)

Fine-tuning: Last 4 layers unfrozen

Accuracy: 99.10%

2. ResNet50
Base Model: Pre-trained on ImageNet

Custom Layers: GlobalAveragePooling2D, BatchNormalization, Dense(256)

Fine-tuning: Last 10 layers unfrozen

Accuracy: 97.69%

3. InceptionV3
Base Model: Pre-trained on ImageNet

Custom Layers: GlobalAveragePooling2D, Dense(512), Dropout(0.4)

Fine-tuning: Last 10 layers unfrozen

Accuracy: 100%
