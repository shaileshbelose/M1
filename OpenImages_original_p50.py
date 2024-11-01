import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Load the reduced images
reduced_images_p50 = np.load('mnist_images/reduced_images_p50.npy')

# Load the MNIST dataset to get the top eigenvectors
mnist = fetch_openml('mnist_784', version=1)
X = mnist["data"][:2000] / 255

# Compute the sample covariance matrix and perform eigendecomposition
mean_vec = np.mean(X, axis=0)
X_centered = X - mean_vec
cov_matrix = np.cov(X_centered, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

# Get the top 50 eigenvectors
top_eigenvectors_p50 = sorted_eigenvectors[:, :50]

# Reconstruct the images from the reduced dimensions
reconstructed_images_p50 = np.dot(reduced_images_p50, top_eigenvectors_p50.T) + mean_vec

# Function to display a grid of reconstructed images
def display_images(images, num_images=10):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.show()

# Display the first 10 reconstructed images in a grid
display_images(reconstructed_images_p50, num_images=10)
