# Hand-Gesture-Recognition-Model
The Hand Gesture Recognition Model is a deep learning project built using Convolutional Neural Networks (CNNs) to classify hand gestures from the LeapGestRecog dataset. The dataset consists of thousands of grayscale images representing different gesture classes. The workflow involves preprocessing the images, splitting them into training and testing sets, and training a CNN model to learn gesture patterns. The trained model can accurately recognize hand gestures, which can be applied in human-computer interaction, sign language translation, gaming, and virtual environments.

📌 Project Overview

This project implements a Hand Gesture Recognition Model using Convolutional Neural Networks (CNNs) to classify hand gestures from images. The model can accurately identify and classify different hand gestures, enabling applications in human-computer interaction, gesture-based control systems, gaming, and virtual reality.

📂 Dataset Description

The dataset used is LeapGestRecog from Kaggle, automatically downloaded using kagglehub
.

The dataset contains 30,000 images of 10 different hand gestures, performed by multiple users.

Each gesture is stored in separate folders (00, 01, ..., 09).

Images are grayscale (120x320 pixels).

Dataset Download (with kagglehub)

You can download the dataset with:

import kagglehub

# Download latest version
path = kagglehub.dataset_download("gti-upm/leapgestrecog")

print("Path to dataset files:", path)


This will create a dataset folder path like:

/root/.cache/kagglehub/datasets/gti-upm/leapgestrecog/versions/1/leapGestRecog/

🔄 Workflow & Methodology

Dataset Loading

Use kagglehub to download LeapGestRecog dataset.

Load images from dataset folders and preprocess (resize, grayscale, normalize).

Model Development (CNN)

Convolution + MaxPooling layers for feature extraction.

Dense + Dropout layers for classification.

Softmax activation for final output.

Training & Evaluation

Train on training set, validate on test set.

Evaluate model accuracy and loss.

Prediction

Load a test image and predict gesture class.

⚙️ Installation

Install dependencies:

pip install tensorflow keras matplotlib numpy opencv-python kagglehub

▶️ How to Run the Project

Clone the repository:

git clone https://github.com/your-username/hand-gesture-recognition.git
cd hand-gesture-recognition


Run the dataset download script:

import kagglehub
path = kagglehub.dataset_download("gti-upm/leapgestrecog")
print("Dataset path:", path)


Open the notebook:

jupyter notebook hand_gesture_recognition.ipynb


Train and test the model.

🧪 Sample Prediction
import cv2, numpy as np

# Load a sample image from dataset
img_path = path + "/leapGestRecog/00/01/palm/frame_01_01_0001.png"

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (64, 64)) / 255.0
img = img.reshape(1, 64, 64, 1)

prediction = model.predict(img)
print("Predicted Gesture:", np.argmax(prediction))

📁 File Structure
hand-gesture-recognition/
│── README.md
│── hand_gesture_recognition.ipynb   # Main notebook
│── requirements.txt                 # Dependencies
│── /models/                         # Saved trained models
│── /results/                        # Graphs & predictions
│── kagglehub_download.py            # Script to download dataset
# Developed By 

HARSHITHA PRASAD S G

GITHUB: harshithaprasadprasad

LINKEDIN : https://www.linkedin.com/in/harshitha-prasad-s-g-55a05a257
