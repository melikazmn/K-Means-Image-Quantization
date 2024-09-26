import os

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to initialize centroids randomly
def initialize_centroids(pixels, k):
    np.random.seed(42)
    random_indices = np.random.choice(pixels.shape[0], k, replace=False)
    centroids = pixels[random_indices]
    return centroids

# Function to assign each pixel to the nearest centroid
def assign_pixels_to_centroids(pixels, centroids):
    distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

# Function to update centroids by calculating the mean of assigned pixels
def update_centroids(pixels, labels, k):
    new_centroids = np.array([pixels[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

# K-Means clustering algorithm
def kmeans(pixels, k, max_iters=100):
    centroids = initialize_centroids(pixels, k)
    for i in range(max_iters):
        labels = assign_pixels_to_centroids(pixels, centroids)
        new_centroids = update_centroids(pixels, labels, k)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, labels

# Function to create the quantized image
def create_quantized_image(pixels, labels, centroids, height, width):
    quantized_pixels = centroids[labels]
    quantized_image = quantized_pixels.reshape(height, width, 3)
    return quantized_image

def calculate_compression_ratio(original_image_path, quantized_image_path):
    original_size = os.path.getsize(original_image_path)
    quantized_size = os.path.getsize(quantized_image_path)
    # Calculate compression ratio
    compression_ratio = original_size / quantized_size
    return compression_ratio

# Read the image using OpenCV
image_path = 'pic.jpg'  # Path to your image
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

# Get image dimensions
height, width, _ = image.shape

# Reshape the image to a 2D array of pixels
pixels = image.reshape(-1, 3)

# Take the number of clusters (k) as input from the user
k = int(input("Enter the number of colors (k): "))

# Apply K-Means clustering to the image pixels
centroids, labels = kmeans(pixels, k)

# Create the quantized image
quantized_image = create_quantized_image(pixels, labels, centroids, height, width)

# Save quantized image
quantized_image_path = f'quantized_image_{k}.png'
quantized_image_bgr = cv2.cvtColor(quantized_image.astype(np.uint8), cv2.COLOR_RGB2BGR)
cv2.imwrite(quantized_image_path, quantized_image_bgr)

# Calculate compression ratio
compression_ratio = calculate_compression_ratio(image_path, quantized_image_path)

# Display the original and quantized images side by side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title(f'Quantized Image with k={k}')
plt.imshow(quantized_image.astype(np.uint8))
plt.axis('off')

print(f'Compression ratio for K={k}: {compression_ratio:.2f}')
plt.show()
