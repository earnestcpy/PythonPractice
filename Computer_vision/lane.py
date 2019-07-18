import cv2

#read image by calling cv2.imread()
load_image = cv2.imread("./Image/Image/test_image.jpg")
cv2.imshow("result", load_image)
cv2.waitKey(0)


