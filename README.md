# Multiclass-Fish-Image-Classification
# 🐟 Multiclass Fish Image Classification

This project focuses on building and deploying a deep learning model to classify different species of fish from images. It uses Convolutional Neural Networks (CNNs) trained from scratch and fine-tuned pretrained models like MobileNetV2 and EfficientNetB0. The final model is deployed using a Streamlit web app.

---

## 📌 Features

- Image classification of multiple fish species.
- Custom CNN and transfer learning with pretrained models.
- Model evaluation using accuracy, precision, recall, F1-score, and confusion matrix.
- Streamlit web app to upload fish images and get predictions with confidence scores.
- Visualizations of training history.

---

## 📊 Dataset

The dataset consists of images of various fish species organized into subfolders per class:
Each image is resized to **224x224** before feeding into the model.

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy / Matplotlib / Seaborn**
- **Streamlit**
- **scikit-learn**

---

## 🧠 Models Used

### 1. CNN from Scratch
A custom-built CNN with multiple Conv2D, MaxPooling2D, and Dropout layers.

### 2. Transfer Learning
- **MobileNetV2**: Pretrained on ImageNet, with top layers fine-tuned.
- **EfficientNetB0**: Lightweight and efficient, also fine-tuned.

---

## 📈 Model Evaluation

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Training vs. Validation Loss and Accuracy Graphs

---

## 🚀 Streamlit Web App

### Features:
- Upload an image.
- Get fish species prediction with confidence scores.
- Visual display of uploaded image and prediction result.



