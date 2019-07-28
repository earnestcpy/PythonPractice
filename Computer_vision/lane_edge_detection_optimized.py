import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_coordinates):
    slope, intercept = line_coordinates
    y1 = image.shape[0]  #y-intercept

    # y = mx + b
    # y2 = line goes three fifth of the way. See lane_edge_detection_optimized_explanation.png
    y2 = int(y1 * (3/5))

    x1 = int((y1- intercept)/slope)
    
    x2 = int((y2- intercept)/slope)

    return np.array([x1,y1,x2,y2])

def average_slope(image, lines):
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        
        #polyfit funtion to find the slope and y intercept for each line in a form of [slope, y-intercept]
        #polyfit(x cooridinates, y cooridninates, degree)   x.shape must be equal to y.shape and degree = 1 indicates to be linear function
        parameters = np.polyfit((x1,x2), (y1,y2),1)
        slope = parameters[0]
        intercept = parameters[1]
        print(parameters)

        if slope < 0: 
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))
    #np.average to return the average slope and intercept for left lanes and right lanes respectively.
    left_average_slope_intercept = np.average(left_lines, axis=0)
    right_average_slope_intercept = np.average(right_lines, axis=0)
    print(left_average_slope_intercept, " : Average slope and intercept for left lane")
    print(right_average_slope_intercept, " : Average slope and intercept for right lane and intercept")

    left_line = make_coordinates(image, left_average_slope_intercept)
    right_line = make_coordinates(image, right_average_slope_intercept)

    return np.array([left_line, right_line])



def cannyFunction(image):
    g_image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    
    #ksize (5,5) could be any odd number. Even number will not be accepted. The higher ksize parameters, The more blurrer the image it is  
    blur_image = cv2.GaussianBlur(g_image, (5, 5), 0)
    
    canny_image = cv2.Canny(blur_image, 50, 150)
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
    mask = np.zeros_like(image) ### important! mask//mask_image need to be set to zeros in order to blend lines_image to original image because bitwise_add() to 0 is equal to zeros    
    #specify the area//polygon needs to be filled in color 255; 
    # mask = image,
    # triangle = Array of polygons where each polygon is represented as an array of points.
    # colorPicker = 255
    cv2.fillPoly(mask, triangle, 255) 
    mask_image = cv2.bitwise_and(image, mask)
    return mask_image
    

def display_lines(image, lines):
    #define a image variable in a form of 3-D array (row, column, colorPicker in heximal and fill with 0 which is black)
    lines_image = np.zeros_like(image) 
    
    if lines is not None: #Null checked
        for line in lines:
            print("Line\'s coordinates:",  line)  
            #e.g [[704 418 927 641]]
            #e.g [[767 493 807 534]]
            #each line is a 2D array containing our line coordinates in a form [[x1, y1, x2, y2]].
            #These coordinates specify the line's parameters, as well as the location of the lines with respect to the image space (in pixel resolution (xxx * yyy)
            x1, y1, x2, y2 = line.reshape(4)

            #(255,0,0) which is blue in ColorPicker in RGB format 
            #1 is the thickness of the lines in the image
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), 10) 

    return lines_image


#read image by calling cv2.imread()
load_image = cv2.imread("./Image/Image/test_image.jpg")

#copy load_image variable over copy_image as a multi-dimensional array
copy_image = np.copy(load_image)

#Call cannyFuntion to proceed the image for edge detection
canny_image = cannyFunction(copy_image)

#Region of interest 
cropped_image = regionOfInterest(canny_image)

#Edge detection   
#minLineLength=40  If traced edge is less than 40 pixel, the line would not be accepted.
#maxLineGap=5  It indeicates the maximum distance in pixels between segmented lines which we will allow to be connected instead of being broken up
#e.g  --- --  to be -----      
edge = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

average_lines = average_slope(copy_image, edge)

#display_lines function draws lines on a black image which is image variable in a form of 3-D array
lines_image = display_lines(copy_image, average_lines)



#Display image with matplotlib.pyplot.imshow() along with x and T axis
# plt.imshow(canny_image)
# plt.show()


#addWeighted(source_image_1, P1, source_image_2, p2, gamma)  
# All elements in source_image_1 will be mutiplied by P1 to decrease the pixel intensity values in image_1, which makes the image_1 a bit darker (optional).
# All elements in source_image_2 will be mutiplied by P2 to decrease the pixel intensity values in image_1, in which makes no difference to the original source_image_2 (no need to dark the lines)
# Lastly gamma will be added to the sum of source_image_1 and source_image_2 
combined_image = cv2.addWeighted(copy_image, 0.8, lines_image, 1, 1)

#Display image with applying simple edge detection
cv2.imshow("result", combined_image)


#cv2.imshow("result", cropped_image)
cv2.waitKey(0)


