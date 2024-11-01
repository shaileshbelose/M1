import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Load the reduced images
reduced_images_p50 = np.load('mnist_images/reduced_images_p50.npy')

# Load the MNIST dataset to get the original data and compute the principal components
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

# Reconstruct the images from the reduced dimensions using the top 50 eigenvectors
mean_vec_array = mean_vec.to_numpy()  # Convert mean_vec to a NumPy array
reconstructed_images_p50 = np.dot(reduced_images_p50, top_eigenvectors_p50.T) + mean_vec_array

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

# Optional: Display the original and reconstructed images side by side for comparison
def display_comparison(original_images, reconstructed_images, num_images=10):
    plt.figure(figsize=(20, 10))
    for i in range(num_images):
        # Original images
        plt.subplot(2, num_images, i + 1)
        plt.imshow(original_images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Original Images')
        # Reconstructed images
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title('Reconstructed Images')
    plt.show()

# Display the first 10 original and reconstructed images side by side
display_comparison(X, reconstructed_images_p50, num_images=10)
