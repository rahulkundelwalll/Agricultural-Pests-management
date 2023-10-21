# Agricultural-Pests-management
This project leverages the power of the EfficientNetV2 pre-trained model to create a real-time monitoring system for crop pest detection. Using a dataset from Kaggle that encompasses various agricultural pests, the model is trained to recognize pests such as ants, bees, beetles, caterpillars, and more.
# Crop Pest Detection using EfficientNetV2

## Overview

This project utilizes the EfficientNetV2 pre-trained model for crop pest detection, aiming to assist in real-life pest management in agricultural settings. The model is designed to live-monitor crops, detect pests, and provide insights for timely intervention.

## Dataset

The dataset used for training and testing the model is sourced from Kaggle: [Agricultural Pests Dataset](https://www.kaggle.com/datasets/gauravduttakiit/agricultural-pests-dataset/). It includes various classes of agricultural pests, such as ants, bees, beetles, caterpillars, earthworms, earwigs, grasshoppers, moths, slugs, snails, wasps, and weevils.

## Model Architecture

The model is based on the EfficientNetV2 architecture, with the EfficientNetV2B0 variant serving as the base model. The top layers of the pre-trained model are frozen, and a custom classification head is added for the specific pest classes. L2 regularization is applied to the Dense layer, and the model is compiled using categorical crossentropy loss and the Adam optimizer.

## Training

The model is trained for 5 epochs on the provided training data. Early stopping is implemented with a patience of 2, ensuring training stops when validation accuracy plateaus.

## Inference and Live Monitoring

The trained model is then used for real-time inference on live webcam feeds. The OpenCV library is employed to capture webcam frames, and the model predicts the pest class in each frame. The results are overlaid on the video feed, providing a live monitoring system for crop pest detection.

## How to Use

1. **Clone the Repository:**
   ```
   git clone https://github.com/rahulkundelwalll/Agricultural-Pests-management
   cd Agricultural-Pests-management
   ```

2. **Download Pre-trained Model:**
   Download the pre-trained EfficientNetV2 model and place it in the project directory.

3. **Run the Live Monitoring Script:**
   ```
   python live_monitoring.py
   ```
   This script captures the webcam feed, performs real-time pest detection, and overlays the results on the video.

## Results

The model's effectiveness can be evaluated by observing its performance during live monitoring. Adjustments and fine-tuning can be made based on real-world feedback to improve accuracy and generalization.

Feel free to contribute, raise issues, or adapt the code for your specific use case. Happy monitoring! 🌾🚜
