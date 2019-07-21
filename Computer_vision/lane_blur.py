import cv2
import numpy as np

#read image by calling cv2.imread()
load_image = cv2.imread("./Image/Image/test_image.jpg")

#copy load_image variable over copy_image as a multi-dimensional array
copy_image = np.copy(load_image)

#convert a RGB image to grayscale
g_image = cv2.cvtColor(copy_image,cv2.COLOR_RGB2GRAY)

#Applying blur to make image smooth and reduce noise in the image in order to proceed to Canny method. Unsmooth image would cause unexpected results.
blur_image = cv2.GaussianBlur(g_image, (5, 5), 0)

#Canny method to trace all the sharp gradient changes in the blur image, and therefore to connect all the sharp gradients as a serie of white pixels. 
# If the change of gradiant is low than 50//low threhold, it will not be traced.
# If the change of gradiant is high than 150// high threshold, it will be traced.
# if the change of gradianta is between the low threshold and the high thresgold, it will ONLY be accepted if it is connected to an edge.
canny_image = cv2.Canny(blur_image, 50, 150)

cv2.imshow("result", canny_image)
cv2.waitKey(0)


