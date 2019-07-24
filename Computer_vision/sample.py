import numpy as np
import cv2 
import matplotlib.pyplot as plt



load_image = cv2.imread("./Image/Image/test_image.jpg")


#copy load_image variable over copy_image as a multi-dimension array
copy_image  = np.copy(load_image)

#Convert to greyscale image
grey_image = cv2.cvtColor(copy_image, cv2.COLOR_BGR2GRAY)

#Remove noise using Gurr
blur_image = cv2.GaussianBlur(grey_image, (5,5), 0)


#Canny function to display gradiant of the image
canny_image = cv2.Canny(blur_image, 50, 150)


cv2.imshow("result", canny_image)
cv2.waitKey(0)