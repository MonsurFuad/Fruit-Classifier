
# 🍎 Fruit Classifier Using Deep Learning (MobileNetV2)

A lightweight and accurate image classification project using deep learning and transfer learning to classify fruit images. This was developed as part of a Computer Vision and Image Processing course at Uttara University.

---

## 📌 Overview

This project leverages MobileNetV2, a pretrained convolutional neural network, to classify images of five types of fruits:

- Apple
- Banana
- Mango
- Grape
- Strawberry

The model is trained on 10,000 images and demonstrates high performance on validation and test data.

---

## 🧠 Model Architecture

- Base Model: MobileNetV2 (frozen, pretrained on ImageNet)
- Custom Layers:
  - GlobalAveragePooling2D
  - Dense(128, activation='relu')
  - Dropout(0.3)
  - Dense(5, activation='softmax')

---

## 📂 Dataset

- Total Images: 10,000
  - Training Set: 9,700 images
  - Validation Set: 100 images
  - Testing Set: 200 images
- Image Size: Resized to 224x224 pixels
- Preprocessing:
  - Normalization (rescale=1./255)
- Augmentation:
  - Rotation (±20°)
  - Width and height shift
  - Zoom
  - Shear
  - Horizontal flip

---

## ✅ Results

| Metric              | Value     |
|---------------------|-----------|
| Training Accuracy   | 97.10%    |
| Testing Accuracy | 92.00%    |

---
## 💻 Technologies Used

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- Pillow (for image processing)
- Jupyter Notebook

---

## 📱 Future Enhancements

- Add more fruit categories (e.g., Mango, Grapes)
- Collect and train on real-world images with diverse lighting and occlusion
- Optimize the model for mobile deployment using TensorFlow Lite
- Add object detection to identify multiple fruits in one image
- Develop a user-friendly GUI or mobile app for real-time classification
