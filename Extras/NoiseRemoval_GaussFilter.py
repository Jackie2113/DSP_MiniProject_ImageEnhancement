#Import the necessary libraries 
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 

# Load the image 
image = cv2.imread('GFG.jpeg') 

#Plot the original image 
plt.subplot(1, 2, 1) 
plt.title("Original") 
plt.imshow(image) 

# Remove noise using a Gaussian filter 
filtered_image2 = cv2.GaussianBlur(image, (7, 7), 0) 

#Save the image 
cv2.imwrite('Gaussian Blur.jpg', filtered_image2) 

#Plot the blured image 
plt.subplot(1, 2, 2) 
plt.title("Gaussian Blur") 
plt.imshow(filtered_image2) 
plt.show()
