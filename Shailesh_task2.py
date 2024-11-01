import numpy as np
from sklearn.datasets import fetch_openml
import os

# Fetch the MNIST dataset from OpenML
mnist = fetch_openml('mnist_784', version=1)

# Extract the data and target values
X, y = mnist["data"], mnist["target"]

# Use the first 2000 samples for both data and target
X = X[:2000]
y = y[:2000]

# Scale the images to the range [0, 1] by dividing each by 255
X_scaled = X / 255

# Compute the sample covariance matrix
mean_vec = np.mean(X_scaled, axis=0)
X_centered = X_scaled - mean_vec
cov_matrix = np.cov(X_centered, rowvar=False)

# Perform eigendecomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Calculate the percentage of variance explained
total_variance = np.sum(eigenvalues)
variance_explained = eigenvalues / total_variance
cumulative_variance_explained = np.cumsum(variance_explained)

# Apply PCA via eigendecomposition to reduce the dimensionality
p_values = [50, 250, 500]
X_reduced_dict = {}

# Sort the eigenvalues and corresponding eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
sorted_eigenvectors = eigenvectors[:, sorted_indices]

for p in p_values:
    # Select the top p eigenvectors
    top_eigenvectors = sorted_eigenvectors[:, :p]
    
    # Project the data onto these eigenvectors to reduce the dimensionality
    X_reduced = np.dot(X_centered, top_eigenvectors)
    X_reduced_dict[p] = X_reduced

    # Print shape of the reduced data for verification
    print(f"Reduced data shape for p={p}: {X_reduced.shape}")

# Create a directory to save the images if it doesn't exist
os.makedirs('mnist_images', exist_ok=True)

# Save the original images
np.save('mnist_images/original_images.npy', X)

# Save the processed images for each p
for p in p_values:
    np.save(f'mnist_images/reduced_images_p{p}.npy', X_reduced_dict[p])

print("Images saved successfully!")
