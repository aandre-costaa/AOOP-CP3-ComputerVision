
# Real-Time American Sign Language (ASL) to Text Translator  
**AOOP - CP3 - Computer Vision Project**  
**Developed by:** André Costa, Nº27638  

---

## Overview

This application provides a real-time translation of American Sign Language (ASL) into text using a custom-trained machine learning model. It leverages **MediaPipe** for precise hand landmark detection and **TensorFlow/Keras** for gesture classification. It supports the full alphabet (A–Z) and digits (0–9), allowing users to build words and phrases using simple hand gestures.

---

## Features

- Real-time webcam input.
- Hand landmark detection using **MediaPipe**.
- Trained deep learning model with **SeparableConv2D** and **Global Average Pooling** for accurate predictions.
- Interactive interface with keyboard shortcuts for text manipulation.

---

## How to Run the Application

1. **Install Dependencies**

   Open a terminal in the project directory ````/CP3-ComputerVision```` and run:

   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Application**

   In the same terminal, run the application:

   ```bash
   python RealTimeDetection.py
   ```

---

## Controls

Use the following keys while the application is running:

| Key         | Action                                 |
|-------------|----------------------------------------|
| `C`         | Clear the current word/phrase          |
| `Space`     | Add a space between words              |
| `Backspace` | Remove the last written character      |
| `Q`         | Quit the application                   |

---

## Reference Image

You can use the image below as a reference for the American Sign Language alphabet to test the application:

[![ASL Alphabet](https://cdn8.bigcommerce.com/s-dc9f5/product_images/uploaded_images/asl-abc-poster.jpg)](https://cdn8.bigcommerce.com/s-dc9f5/product_images/uploaded_images/asl-abc-poster.jpg)

---

## Training dataset

The training dataset used to fully train the model used can be found in the following page:
https://www.kaggle.com/datasets/vignonantoine/mediapipe-processed-asl-dataset

---

## Dependencies

Ensure that the following libraries are installed (included in `requirements.txt`):

- `tensorflow`
- `mediapipe`
- `opencv-python`
- `numpy`
