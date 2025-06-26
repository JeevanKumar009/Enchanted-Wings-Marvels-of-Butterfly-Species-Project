# Enchanted-Wings-Marvels-of-Butterfly-Species-Project

An AI-powered tool to classify butterfly species from images using deep learning and transfer learning techniques. This project combines the power of computer vision with ecological research to assist in biodiversity monitoring, ecological studies, and citizen science.

## 🌟 Overview

This project leverages a pre-trained Convolutional Neural Network (VGG16) to accurately classify butterfly species. It uses a dataset of **75 butterfly classes** and **6,499 images**, split into training, validation, and test sets. The final model is deployed using a simple yet powerful **Flask web application** where users can upload butterfly images and get species predictions in real-time.

## 📌 Key Features

- ✅ Transfer learning with VGG16 for efficient training
- ✅ Real-time butterfly species identification
- ✅ Web-based user interface for easy access
- ✅ Lightweight model suitable for local or server deployment
- ✅ Enhances scientific research, education, and conservation efforts

## 📂 Project Structure

```

.
├── app.py                     # Flask web app
├── train\_model.py            # Model training script using VGG16
├── vgg16\_model.h5            # Trained model file
├── class\_indices.json        # Mapping of class indices to species names
├── /templates                # HTML templates for web interface
├── /static/uploads           # Uploaded images storage
└── /dataset/train            # Training data organized by class

````

## 🧠 How It Works

### 1. Training
- Dataset loaded using `ImageDataGenerator` with train/validation split
- VGG16 (pre-trained on ImageNet) used as the base model
- Custom dense layers added for butterfly classification
- Model trained for 10 epochs and saved as `vgg16_model.h5`

### 2. Deployment
- A Flask app loads the model and accepts image uploads
- Images are resized to 224x224 and normalized
- The model predicts the species and returns results to the user

## 🔬 Use Case Scenarios

- **Biodiversity Monitoring**: Field researchers can identify species in real-time.
- **Ecological Research**: Track migration, habitat, and behavior.
- **Citizen Science & Education**: Enable enthusiasts to learn and participate in conservation efforts.

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- TensorFlow
- Flask
- NumPy
- Pillow

### Run the Web App

```bash
python app.py
```

Visit `http://127.0.0.1:5000/` in your browser to use the web interface.

## 🙌 Acknowledgements

* Butterfly dataset from [Kaggle](https://www.kaggle.com/datasets/).
* Pre-trained VGG16 model from Keras applications.
* Inspired by efforts in ecological AI and biodiversity monitoring.



