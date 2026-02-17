import os
import argparse
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from polySim import generate_structure as pm
from skimage.filters import laplace, gaussian
from skimage.util import img_as_ubyte as bit

def microsimulator(size=512):
    """Generates a base polycrystalline microstructure using polySim."""
    nucleation_rate = random.randint(10, 20)
    growth_rate = random.randint(1, 3)
    img = np.zeros((size, size), dtype=np.uint16)
    
    # Run the polySim simulation
    # parameters: img, nucleationRate, growthRate, height/steps
    pm(img, nucleation_rate, growth_rate, 300)
    return img

def apply_random_blur_mask(image):
    """
    Applies a heavy gaussian blur to a random crop of the image 
    to simulate discontinuities or 'broken' boundaries.
    """
    img_size = image.shape[0]
    # Scale the blur spot size relative to the image size
    min_dim = max(5, int(img_size * 0.02))
    max_dim = max(10, int(img_size * 0.08))
    dim = random.randint(min_dim, max_dim)
    
    x = random.randint(0, img_size - dim)
    y = random.randint(0, img_size - dim)
    
    crop = image[x:x+dim, y:y+dim]
    # Apply heavy sigma to essentially erase the line in that region
    gau = gaussian(crop, sigma=20, mode='nearest', cval=0)
    gau = bit(gau)
    
    image[x:x+dim, y:y+dim] = gau
    return image

def generate_dataset(output_path, num_images, img_size, start_idx=0):
    """Main loop to generate and save paired images."""
    micro_dir = os.path.join(output_path, "micro")
    label_dir = os.path.join(output_path, "label")
    
    os.makedirs(micro_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    print(f"Starting generation of {num_images} images with resolution {img_size}x{img_size}...")
    print("Output format: Black grain boundaries (0) on a White background (255).")

    for i in range(start_idx, start_idx + num_images):
        # 1. Generate base grain structure
        base_img = microsimulator(size=img_size)
        
        # 2. Get grain boundaries (Ground Truth)
        trace = laplace(base_img, ksize=3)
        trace[trace != 0] = 255
        trace = trace.astype(np.uint8)
        ground_truth = np.copy(trace)
        
        # 3. Create broken boundaries (Input Micrograph)
        broken_trace = np.copy(trace)
        # Scale the number of blur spots based on image resolution
        num_blur_spots = random.randint(int(30 * (img_size/512)), int(100 * (img_size/512)))
        for _ in range(num_blur_spots):
            broken_trace = apply_random_blur_mask(broken_trace)
        
        # 4. Post-processing (Thresholding & Dilation)
        _, input_micro = cv2.threshold(broken_trace, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Add consistent dilation to mimic thickness
        kernel_size = random.randrange(1, 4, step=2)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        input_micro = cv2.dilate(input_micro, kernel, iterations=1)
        ground_truth = cv2.dilate(ground_truth, kernel, iterations=1)

        # 5. Reverse Foreground/Background
        # Current: 255 = boundary, 0 = background
        # Target: 0 = boundary, 255 = background
        input_micro = 255 - input_micro
        ground_truth = 255 - ground_truth

        # 6. Save images
        cv2.imwrite(os.path.join(micro_dir, f"{i}.png"), input_micro)
        cv2.imwrite(os.path.join(label_dir, f"{i}.png"), ground_truth)

        if (i - start_idx + 1) % 10 == 0 or (i - start_idx + 1) == num_images:
            print(f"Generated {i - start_idx + 1}/{num_images} images...")

    print(f"Generation complete. Files saved in {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Synthetic Broken Grain Boundary Dataset")
    parser.add_argument("--output", type=str, default="./generated_data", help="Path to save the dataset")
    parser.add_argument("--count", type=int, default=10, help="Number of image pairs to generate")
    parser.add_argument("--size", type=int, default=512, help="Resolution of images (e.g., 512 for 512x512)")
    parser.add_argument("--start", type=int, default=0, help="Starting index for file naming")
    
    args = parser.parse_args()
    generate_dataset(args.output, args.count, args.size, args.start)
