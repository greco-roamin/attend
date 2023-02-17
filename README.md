# Temporal Attention
Code from the paper titled "Temporal Attention Signatures for Interpretable Time-Series Prediction"

Abstract: Deep neural networks are widely used for time-series prediction due to their popularity and accuracy, but their internal workings are often considered inscrutable. Recent progress has been made in the interpretation of neural networks, particularly in models employing attention mechanisms and feature maps for image classification. These models allow for the construction of feature heat maps that elucidate the salient aspects of the data. While human observers can easily verify the importance of features in image classification, time-series data and modeling is more difficult to interpret. To address this issue, we propose a new approach that combines temporal attention - a fusion of recurrent neural networks and attention - with attention visualization to create temporal attention signatures, which are similar to image attention heat maps. We demonstrate that temporal attention, in addition to achieving higher accuracy than recurrence alone, will show that different label classes result in different attention signatures, indicating that neural networks attend to different portions of time-series sequences depending on what is being predicted. We conclude by discussing practical applications of this novel approach, including model interpretation, and assistance in selecting sequence length and model validation, for building more robust, accurate, and high-confidence interpretable models.

## Requirements

Python 3.7
Tensor Flow 2.4

## Configuration

All configurable parameters are controlled from cfg.py

## Usage

Usage: main-tg.py [-vg]
       (assuming python3 in /usr/bin/)

v: verbose mode (optional)
g: graphing mode (optional)
