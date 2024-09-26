# K-Means Image Quantization with Compression Ratio Calculation
This repository contains a Python script that applies K-Means clustering to reduce the number of colors in an image, a process known as image quantization. It also calculates the compression ratio between the original and quantized images, providing a comparison of file sizes after color reduction.

## Features

- Image Loading and Preprocessing: The script uses OpenCV to load images, which are then reshaped into a 2D array of pixel values.
- K-Means Clustering: Users can input the number of clusters (k), where each cluster represents a distinct color. The script uses the K-Means algorithm to group the pixel colors into k clusters and assign each pixel to the nearest cluster centroid.
- Image Quantization: The pixel values are replaced with the corresponding centroid colors, creating a quantized image with reduced color complexity.
- Compression Ratio Calculation: The script compares the file sizes of the original and quantized images and computes the compression ratio.
- Visualization: Displays the original and quantized images side-by-side using Matplotlib.

## How It Works

- Centroid Initialization: Centroids are randomly selected from the image's pixel values.
- Pixel Assignment: Each pixel is assigned to the closest centroid based on Euclidean distance.
- Centroid Update: Centroids are updated to the mean of the assigned pixel values.
- Repeat: The process repeats until the centroids no longer change or the maximum number of iterations is reached.
- Compression Calculation: The sizes of the original and quantized images are compared to compute the compression ratio.

## Requirements
- Python 3.x
- OpenCV
- NumPy
- Matplotlib

## Example
If you input k = 8, the script will:
- Quantize the image to 8 colors.
- Display the original and quantized images.
- Save the quantized image as quantized_image_8.png.
- Print the compression ratio in the terminal, comparing the original image size to the quantized one.

## Compression Ratio
- The compression ratio is calculated as:
 `Compression Ratio = (Original Image Size) / (Quantized Image Size)`
A higher ratio means greater compression.
