import cv2
import numpy as np
import matplotlib.pyplot as plt
import Convolution as co

#Converting bgr to greyscale
src = cv2.imread('Highimgnoise.png')
imgC = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
imgG = cv2.cvtColor(imgC, cv2.COLOR_RGB2GRAY)

#Comparing the images
plt.subplot(2,2,1)
plt.axis("off")
plt.title("1.BGR")
plt.imshow(src)

plt.subplot(2,2,2)
plt.axis("off")
plt.title("2.RGB")
plt.imshow(imgC)

plt.subplot(2,2,3)
plt.axis("off")
plt.title("3.Grayscale")
plt.imshow(imgG)

#Changing colormap of MATLAB Plot
plt.subplot(2,2,4)
plt.axis("off")
plt.title("4.Grayscale(fixed)")
plt.imshow(imgG, cmap = plt.cm.gray)

#Uncomment below code to view the difference
#plt.show()

#Saving the converted image[uncomment the below code to save]
#cv2.imwrite('selfie_rgb.jpg', img)

#Converting image to a matrix
img_mat = co.convert_image_matrix('Highimgnoise.png')

#Different Convolution kernels
identity_kernel = np.array([[0,0,0],[0,1,0],[0,0,0]])
edgeDet_kernel = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
gaussBlur_kernel = 0.0625 * np.array([[1,2,1],[2,4,2],[1,2,1]])
boxBlur_kernel = 0.1111 * np.array([[1,1,1],[1,1,1],[1,1,1]])

img_sampling = co.get_sub_matrices(img_mat, boxBlur_kernel.shape)
transform_mat = co.get_transformed_matrix(img_sampling, boxBlur_kernel)
co.original_VS_convoluted('Highimgnoise.png','Box Blur', transform_mat)