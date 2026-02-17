# **Reconstruction-GB**

This repository contains the Python implementation for generating synthetic single-phase polycrystalline micrographs with programmatically "broken" or discontinuous grain boundaries. This tool is designed to create high-quality, pixel-perfect ground truth datasets for training and benchmarking deep learning models in microstructural reconstruction and segmentation.

## **Overview**

Automated grain boundary detection often fails in real-world scenarios due to poor etching, low contrast, or imaging artifacts. This script simulates these challenges programmatically by:

1. **Simulating Grain Growth:** Using the polySim library to create realistic polycrystalline templates.  
2. **Extracting Boundaries:** Applying Laplace filters to obtain precise grain boundary skeletons.  
3. **Simulating Discontinuities:** Applying localized Gaussian blurring and thresholding to "break" segments of the boundaries.  
4. **Standardized Output:** Exporting images as **black grain boundaries (0)** on a **white background (255)**.

## **The polySim Library**

A core component of this project is the **polySim** library, an in-house tool developed by our research group. It is exclusively used here to simulate the fundamental nucleation and growth processes of polycrystalline materials. By leveraging polySim, we can generate an infinite variety of microstructural morphologies with known topological properties, providing the essential "clean" templates required for creating our degraded training pairs.

## **Installation**

### **Prerequisites**

* Python 3.8 or higher  
* pip (Python package installer)

### **Detailed Installation Steps**

To ensure there are no conflicts with other Python projects, we recommend installing the dependencies in a virtual environment.

1. **(Optional) Create a Virtual Environment:**  
   python \-m venv venv  
   source venv/bin/activate  \# On Windows use: venv\\Scripts\\activate

2. **Upgrade pip:** Ensure you have the latest version of pip to avoid installation issues:  
   python \-m pip install \--upgrade pip

3. **Install polySim from PyPI:** The polySim library developed by our group is available on PyPI and can be installed directly using:  
   pip install polySim

   *For more details, visit the official PyPI page: [https://pypi.org/project/polySim/](https://pypi.org/project/polySim/)*  
4. **Install Other Required Dependencies:** After installing polySim, install the remaining libraries needed for the generation scripts:  
   pip install numpy opencv-python matplotlib scikit-image

## **Usage**

You can generate a custom dataset by running the generate\_data.py script from your terminal.

### **Basic Command:**

python generate\_data.py \--count 100 \--size 512 \--output ./my\_dataset

### **Arguments:**

* \--count: The total number of image pairs (Input/Label) you wish to generate.  
* \--size: The resolution of the square images (e.g., 512 for 512x512).  
* \--output: The directory path where the micro and label folders will be created.  
* \--start: The starting integer for file naming.

## **Data Structure**

The script produces two synchronized folders:

* /micro: Input images with broken/discontinuous boundaries (Black on White).  
* /label: Ground truth images with fully connected, original boundaries (Black on White).

## **Citation**

If you use this code, the polySim library, or the resulting dataset in your research, please cite:

**Library Reference:**

N. Chaurasia, S.K. Jha, S. Sangal, "A novel training methodology for phase segmentation of steel microstructures using a deep learning algorithm," *Materialia* 30 (2023) 101803\.

## **License**

This project is licensed under the MIT License.