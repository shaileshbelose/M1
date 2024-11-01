import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.metrics import peak_signal_noise_ratio

# Load original images (assuming you have original_images.npy)
original_images = np.load('mnist_images/original_images.npy')

# Perform PCA with different numbers of components
p_values = [50, 250, 500]
reconstructed_images = {}

for p in p_values:
    # Fit PCA and transform original images
    pca = PCA(n_components=p)
    reduced_images = pca.fit_transform(original_images)
    
    # Reconstruct images from reduced data
    reconstructed_images[p] = pca.inverse_transform(reduced_images)

    # Check shape of reconstructed images
    print(f"Shape of reconstructed images (p={p}):", reconstructed_images[p].shape)

# Calculate PSNR values for each reconstruction
psnr_values = {}

for p in p_values:
    psnr_values[p] = []
    for i in range(len(original_images)):
        psnr = peak_signal_noise_ratio(original_images[i].reshape(28, 28), reconstructed_images[p][i].reshape(28, 28))
        psnr_values[p].append(psnr)

# Print PSNR values
for p in p_values:
    print(f"PSNR values for p={p}:")
    for i, psnr in enumerate(psnr_values[p], 1):
        print(f"Image {i}: PSNR = {psnr:.2f} dB")

# Optionally, you can plot the PSNR values
plt.figure(figsize=(10, 6))
for p in p_values:
    plt.plot(psnr_values[p], label=f"p={p}")
plt.xlabel('Image Index')
plt.ylabel('PSNR (dB)')
plt.title('PSNR Comparison for PCA Reconstruction')
plt.legend()
plt.grid(True)
plt.show()
