# Face-Expression-Detection

This project implements a deep learning model to detect and classify facial expressions into seven emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training Process](#training-process)
- [GUI Application](#gui-application)
- [Future Improvements](#future-improvements)
- [Results](#results)

## Overview

This project uses Convolutional Neural Networks (CNNs) to detect human facial expressions from images. The model is trained on a dataset of grayscale facial images and can classify expressions into seven different emotional states. A graphical user interface is provided for easy interaction with the trained model.

## Features

- Classification of facial expressions into 7 emotion categories
- Custom CNN architecture with batch normalization and dropout layers
- Interactive GUI for uploading and analyzing images
- Real-time emotion prediction on uploaded images

## Requirements

- Python 3.6+
- TensorFlow 2.x
- Pandas
- NumPy
- Matplotlib
- OpenCV (cv2)
- PIL (Pillow)
- tkinter
- scikit-learn

## Project Structure

```
facial-expression-recognition/
│
├── train/                            # Training dataset directory
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
├── test/                             # Testing dataset directory
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
│
├── model_training.py                 # Script for model training
├── emotion_detector_app.py           # GUI application script
├── model_a.json                      # Saved model architecture
├── model_weights.weights.h5          # Saved model weights
├── haarcascade_frontalface_default.xml  # Face detection cascade file
└── README.md                         # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kurakula-prashanth/face-expression-detection.git
   cd facial-expression-recognition
   ```

2. Install the required packages:
   ```bash
   pip install tensorflow pandas numpy matplotlib opencv-python pillow scikit-learn
   ```

3. Organize your dataset in the directory structure shown above or adjust the paths in the code accordingly.

## Usage

### Training the Model

To train the model on your dataset:

```bash
python model_training.py
```

This will:
- Load images from the train/ and test/ directories
- Preprocess the images (resize to 48x48 pixels and convert to grayscale)
- Train the CNN model
- Save the model architecture and weights

### Using the GUI Application

To use the trained model with the GUI application:

```bash
python emotion_detector_app.py
```

1. Click on "Upload Image" to select an image from your computer
2. Click on "Detect Emotion" to analyze the facial expression in the image
3. The predicted emotion will be displayed below the image

## Model Architecture

The CNN architecture consists of:

1. Four convolutional blocks, each containing:
   - Convolutional layer
   - Batch normalization
   - ReLU activation
   - Max pooling
   - Dropout (0.25)

2. Flatten layer to convert the 2D feature maps to 1D feature vectors

3. Two dense blocks, each containing:
   - Fully connected layer
   - Batch normalization
   - ReLU activation
   - Dropout (0.25)

4. Output layer with softmax activation for 7-class classification

The model is compiled with:
- Optimizer: Adam (learning rate = 0.0005)
- Loss function: Categorical cross-entropy
- Metric: Accuracy

## Training Process

The model is trained with the following parameters:
- Image size: 48x48 pixels (grayscale)
- Batch size: 64
- Epochs: 15
- Callbacks:
  - ModelCheckpoint: Saves the best model based on validation accuracy
  - ReduceLROnPlateau: Reduces learning rate when validation loss plateaus

## GUI Application

The application uses:
- tkinter for the graphical user interface
- OpenCV's Haar Cascade classifier for face detection
- The trained CNN model for emotion classification

The interface allows users to:
1. Upload an image
2. Detect faces in the image
3. Classify the emotion displayed in each detected face

## Future Improvements

- Real-time emotion detection using webcam feed
- Multi-face detection and emotion recognition
- Model optimization for faster inference
- Additional emotions or more granular emotion detection
- Deployment as a web application

## Results
![image](https://github.com/kurakula-prashanth/Face-Expression-Detection/assets/144904506/336aff59-5765-4525-9e5d-404cfc550ce3)
![image](https://github.com/kurakula-prashanth/Face-Expression-Detection/assets/144904506/1b428546-76e2-4234-8ec4-c0fde0314156)
![image](https://github.com/kurakula-prashanth/Face-Expression-Detection/assets/144904506/7471af1e-8d2e-4736-b7a4-28ea5fa19f68)
