import cv2
import numpy as np
import matplotlib.pyplot as plt



def cannyFunction(image):
    g_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(image, (5, 5), 0)
    canny_image = cv2.Canny(image, 50, 150)
    return canny_image

def regionOfInterest(image): 
    height = image.shape[0]

    #specifying the region where we want to isloate from the image
    triangle = np.array([
        [(200, height), (1100, height), (600, 250)]
    ])
    # print(triangle.shape) //output = (1,3,2)
    # print(triangle[0][2][0])
    # triangle[0][0][0] = 200
    # trinagle[0][0][1] = 700 //height of the given image
    # triangle[0][1][0] = 1100
    # triangle[0][1][1] = 700 //height of the given image
    # triangle[0][2][0] = 600
    # triangle[0][2][1] = 250
    mask = np.zeros_like(image)
    
    #specify the area//polygon needs to be filled in color 255; 
    # mask = image,
    # triangle = Array of polygons where each polygon is represented as an array of points.
    # colorPicker = 255
    cv2.fillPoly(mask, triangle, 255) 
    mask_image = cv2.bitwise_and(image, mask)
    return mask_image
    
#read image by calling cv2.imread()
load_image = cv2.imread("./Image/Image/test_image.jpg")

#copy load_image variable over copy_image as a multi-dimensional array
copy_image = np.copy(load_image)

#Call cannyFuntion to proceed the image for edge detection
canny_image = cannyFunction(copy_image)

#display image with openCV.imshow
# cv2.imshow("result", canny_image)
# cv2.waitKey(0)

#Display image with matplotlib.pyplot.imshow() along with x and T axis
# plt.imshow(canny_image)
# plt.show()

cv2.imshow("result", regionOfInterest(canny_image))
cv2.waitKey(0)


