# Temporal Attention
Code from the paper titled "Temporal Attention for Improved Time Series
Classification and Interpretability" in review with Artificial Intelligence
in Medicine, AIIM 2022

Abstract: With the advent of transformer deep learning architectures, recurrent neural networks have subsided in popularity in research areas related to text and image processing. Attention-based models have been utilized to demystify the ``black-box'' nature of neural networks through the construction of heat maps identifying what the network considers important during classification. One disadvantage of pure attention-only models is that they ignore the order and temporal proximity of time steps in time series data. Although input order can be encoded using positional encoding, the effect of temporal proximity is not directly modeled. Recurrent networks, on the other hand, inherently maintain a memory of recent time steps. Previous work has shown that combining recurrence with attention can exceed the accuracy of recurrence alone. In this work, we show that recurrence combined with temporal attention can exceed transformer performance (i.e., self-attention alone). We show that self-attention can work together with recurrence to achieve high accuracy, and regular attention can be utilized to generate \emph{temporal attention signatures} to increase the interpretability of time series classification by applying the concept of the visual attention heat map to time-series data. Temporal attention signatures, analogous to visual attention heat-maps, clearly indicate that different classes typically produce different signatures and yield practical insight into interpreting, building, and validating medical and natural process time-series models.

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
