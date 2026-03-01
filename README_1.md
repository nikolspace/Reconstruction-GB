# **Reconstruction-GB: Synthetic Micrograph Generation & Deep Learning Framework**

This repository provides the complete programmatic methodology and deep learning framework for the paper: **"From Fragmented to Flawless: A Large-Scale Synthetic Micrograph Library for Benchmarking Microstructural Image Restoration"**.

## **Overview**

Automated grain boundary analysis is often hindered by experimental artifacts. This project provides:

1. **Programmatic Data Generation:** Algorithms to create synthetic micrographs with broken boundaries, etching pits, and scratches using the polySim library.  
2. **Deep Learning Baseline:** An **Attention U-Net** implementation for microstructural reconstruction and denoising.

## **The polySim Library**

A core component of this project is the **polySim** library, an in-house tool developed by our group to simulate physical nucleation and growth processes.

* **PyPI:** [https://pypi.org/project/polySim/](https://pypi.org/project/polySim/)  
* **Installation:** pip install polySim

## **Installation**

Ensure you have Python 3.8+ installed. We recommend using a virtual environment.

\# Install dependencies  
pip install polySim numpy opencv-python matplotlib scikit-image tensorflow keras sklearn

## **1\. Dataset Generation**

The script generate\_data.py allows you to create your own version of the 14,999 image pairs.

**Task: Reconstruction (Broken Boundaries)**

python generate\_data.py \--task reconstruction \--count 100 \--size 512

**Task: Denoising (Pits & Scratches)**

python generate\_data.py \--task denoising \--count 100 \--size 512

## **2\. Training and Testing**

The script train\_reconstruction.py implements an **Attention U-Net** to restore broken grain boundaries.

\# Start training (Point to your generated data folders)  
python train\_reconstruction.py \--data\_dir ./generated\_data/reconstruction \--epochs 100 \--batch\_size 4

### **Model Architecture**

The provided code implements an Attention-Gated U-Net. The Attention Gate highlights relevant feature maps and suppresses irrelevant regions (noise), which is critical for connecting discontinuous grain boundary segments.

## **Citation**

If you use this code or dataset, please cite:

N. Chaurasia, S. Sangal, S.K. Jha, "From Fragmented to Flawless: A Large-Scale Synthetic Micrograph Library for Benchmarking Microstructural Image Restoration," *Data in Brief* (2025).

## **License**

MIT License \- See LICENSE file for details.