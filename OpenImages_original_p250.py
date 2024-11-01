import numpy as np
import matplotlib.pyplot as plt

# Load the original images
original_images = np.load('mnist_images/reduced_images_p250.npy')

# Define the dimensions of the images
img_height, img_width = 28, 28

# Reshape the first image to 28x28 pixels and display it
first_image = original_images[0].reshape(img_height, img_width)

# Plot the first image
plt.figure(figsize=(4, 4))
plt.imshow(first_image, cmap='gray')
plt.title('First Original Image')
plt.axis('off')
plt.show()

# Function to display a grid of images
def display_images(images, num_images=10):
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].reshape(img_height, img_width), cmap='gray')
        plt.axis('off')
    plt.show()

# Display the first 10 images in a grid
display_images(original_images, num_images=10)
