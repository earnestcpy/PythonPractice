import cv2
import numpy as np

#read image by calling cv2.imread()
load_image = cv2.imread("./Image/Image/test_image.jpg")

#copy load_image variable over copy_image as a multi-dimensional array
copy_image = np.copy(load_image)

#convert a RGB image to grayscale
g_image = cv2.cvtColor(copy_image,cv2.COLOR_RGB2GRAY)


cv2.imshow("result", g_image)
cv2.waitKey(0)


