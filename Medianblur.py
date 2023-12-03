import cv2
import numpy as np

input_image_path = 'Noise_salt_and_pepper.png'
output_image_path = 'Noise_salt_and_pepper_medBlur.png'
kernel_size = 5  

def median_blur_custom(image, kernel_size):
    height, width = image.shape
    half_kernel = kernel_size // 2
    result = np.zeros((height, width), dtype=np.uint8)

    for i in range(half_kernel, height - half_kernel):
        for j in range(half_kernel, width - half_kernel):
            # Extract the local neighborhood
            neighborhood = image[i - half_kernel : i + half_kernel + 1, j - half_kernel : j + half_kernel + 1]

            # Calculate the median value and assign it to the result image
            result[i, j] = np.median(neighborhood)

    return result

img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
blurred_custom = median_blur_custom(img, kernel_size)
cv2.imwrite(output_image_path, blurred_custom)
