import matplotlib.pyplot as plot
import numpy as np
from sklearn.datasets import fetch_openml


#Needs to Fetch the MNIST dataset from OpenML  and then select top 2000 for test data
mnist = fetch_openml('mnist_784', version=1)

# Extract the data and target values
X, y = mnist["data"], mnist["target"]

# Use the first 2000 samples for both data and target
X = X[:2000]
y = y[:2000]

# cale the images to the range [0, 1] by dividing each by 255 as per the first step given.
X_scaled = X / 255

# Compute the sample covariance matrix as step 2
mean_vec = np.mean(X_scaled, axis=0)
X_centered = X_scaled - mean_vec
cov_matrix = np.cov(X_centered, rowvar=False)

# Perform eigendecomposition step 3
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Calculate the percentage of variance explained step 4
total_variance = np.sum(eigenvalues)
variance_explained = eigenvalues / total_variance
cumulative_variance_explained = np.cumsum(variance_explained)

# Plot the cumulative sum of the percentages versus the number of components step5
plot.figure(figsize=(10, 6))
plot.plot(cumulative_variance_explained * 100)
plot.xlabel('Number of Components')
plot.ylabel('Cumulative Variance Explained (%)')
plot.title('Cumulative Variance Explained by Principal Components')
plot.grid(True)
plot.show()
