# **Reconstruction-GB: Synthetic Micrograph Generation & Deep Learning Framework**

This repository provides the complete programmatic methodology and deep learning framework for the paper: **"From Fragmented to Flawless: A Large-Scale Synthetic Micrograph Library for Benchmarking Microstructural Image Restoration"**.

## **Overview**

Automated grain boundary analysis is a critical prerequisite for quantitative metallography, yet it is often hindered by fragmented or discontinuous boundaries caused by poor etching, low contrast, or imaging artifacts.

This project provides:

1. **Programmatic Data Generation:** Algorithms to create synthetic micrographs with fragmented grain boundaries using the polySim library.  
2. **Deep Learning Framework:** A robust **Attention U-Net** implementation specifically optimized to bridge gaps and restore connectivity in discontinuous grain boundary networks.

**Note:** This repository is dedicated exclusively to the **Grain Boundary Reconstruction** task. A separate library and repository addressing Microstructural Denoising (noise cleaning, etching pits, and scratch removal) will be released in the near future.

## **The polySim Library**

The foundation of this dataset is the **polySim** library, an in-house tool developed by our research group to simulate physical nucleation and growth processes of polycrystalline materials.

* **PyPI:** [https://pypi.org/project/polySim/](https://pypi.org/project/polySim/)  
* **Installation:** pip install polySim

## **Installation**

### **Prerequisites**

* Python 3.8 or higher  
* pip (Python package installer)

### **Setup**

We recommend using a virtual environment to manage dependencies and ensure a clean environment:

\# Create and activate virtual environment  
python \-m venv venv  
source venv/bin/activate  \# On Windows: venv\\Scripts\\activate

\# Install required libraries  
pip install polySim numpy opencv-python matplotlib scikit-image tensorflow keras sklearn

## **1\. Dataset Generation**

The script generate\_data.py allows for the reproduction of the 14,999 image pairs used in our study. It simulates discontinuities in the grain boundary network by applying programmatic Gaussian blur masks to "clean" simulated templates.

### **Usage Example:**

To generate 100 image pairs (Input Micrograph / Ground Truth Label) at 512x512 resolution:

python generate\_data.py \--task reconstruction \--count 100 \--size 512

The data will be organized in ./generated\_data/reconstruction/ under micro and label subdirectories.

## **2\. Training and Testing**

The script train\_and\_test.py provides an end-to-end deep learning workflow using an **Attention-Gated U-Net**. This architecture is specifically chosen because its attention mechanism effectively ignores background noise while focusing on the linear features of fragmented grain boundaries.

### **How to Run:**

To train the model on your generated dataset and evaluate it on the test set:

python train\_and\_test.py \--data ./generated\_data/reconstruction \--epochs 100 \--batch 4 \--limit 5000

### **Evaluation & Metrics:**

Upon completion, the script will:

* Save the best performing weights as best\_reconstruction\_model.hdf5.  
* Calculate the **Mean Intersection over Union (mIoU)** for the test set to quantify pixel-level accuracy.  
* Generate a visualization comparing the fragmented input image, the ground truth, and the model's predicted reconstruction.

## **Technical Details**

For an in-depth explanation of the Attention U-Net architecture and the mathematical methodology behind the boundary fragmentation, please refer to the README\_Extended.md file in this repository.

## **Citation**

If you use this code, the polySim library, or the resulting dataset in your research, please cite:

N. Chaurasia, S. Sangal, S.K. Jha, "From Fragmented to Flawless: A Large-Scale Synthetic Micrograph Library for Benchmarking Microstructural Image Restoration," *Data in Brief* (2025).

## **License**

This project is licensed under the MIT License \- see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.