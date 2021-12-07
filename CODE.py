
import cv2
import numpy as np
import os


def detect_shapes(img):

#     CODE WRITTEN BY ME STARTS HERE
    detected_shapes = []

    ##############	ADD YOUR CODE HERE	##############
    # Converting into Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # setting threshold of gray image
    _, threshold = cv2.threshold(gray, 165, 255, cv2.THRESH_BINARY)
    
    # using a findContours() function
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    i = 0

    lower_red = np.array([0, 150, 50])
    upper_red = np.array([10, 255, 255])

    lower_blue = np.array([115, 150, 0])
    upper_blue = np.array([125, 255, 255])

    lower_green = np.array([45, 150, 50])
    upper_green = np.array([65, 255, 255])

    lower_orange = np.array([15, 150, 0])
    upper_orange = np.array([25, 255, 255])

    grid_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    grid_hsv = cv2.cvtColor(grid_RGB, cv2.COLOR_RGB2HSV)

    mask1 = cv2.inRange(grid_hsv, lower_red, upper_red)

    mask2 = cv2.inRange(grid_hsv, lower_blue, upper_blue)
    
    mask3 = cv2.inRange(grid_hsv, lower_green, upper_green)

    mask4 = cv2.inRange(grid_hsv, lower_orange, upper_orange)

    
    

    #res = cv2.bitwise_and(grid_RGB, grid_RGB, mask=mask)
    
    # Putting Values into list
    for contour in contours:
                # here we are ignoring first counter because
                # findcontour function detects whole image as shape
                if i == 0:
                        i = 1
                        continue
                
                # finding Colour
                
                c=''

                if cv2.countNonZero(mask1) > 0:
                    c='Red'
    

                if cv2.countNonZero(mask2) > 0:
                    c='blue'
    

                if cv2.countNonZero(mask3) > 0:
                    c='green'
                

                else:
                    c='orange'

                # finding Shape
                approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)

                # if the shape is a triangle, it will have 3 vertices
                if ((len(approx) == 3) or (len(approx) == 6)):
                        sh='Triangle'

                # if the shape has 4 vertices, it is either a square or a rectangle
                elif len(approx) == 4:
                        # compute the bounding box of the contour and use the
                        # bounding box to compute the aspect ratio
                        (x, y, w, h) = cv2.boundingRect(approx)
                        ar = w / float(h)

                        # a square will have an aspect ratio that is approximately
                        # equal to one, otherwise, the shape is a rectangle
                        if (ar >= 0.95) and (ar <= 1.05):
                                sh = "square"
                        else:
                                sh = "rectangle"

                # if the shape is a pentagon, it will have 5 vertices
                elif len(approx) == 5:
                        sh='Pentagon'

                # otherwise, we assume the shape is a circle
                else:
                        sh='circle'

                # finding center point of shape
                M = cv2.moments(contour)
                t=()
                if M['m00'] != 0.0:
                        x = int(M['m10']/M['m00'])
                        y = int(M['m01']/M['m00'])
                t=(x,y)
                
                detected_shapes.append([c,sh,t])
                
  
    ##################################################
    
    return detected_shapes

    #     CODE WRITTEN BY ME ENDS HERE

def get_labeled_image(img, detected_shapes):
      

    for detected in detected_shapes:
        colour = detected[0]
        shape = detected[1]
        coordinates = detected[2]
        cv2.putText(img, str((colour, shape)),coordinates, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    return img

if __name__ == '__main__':
    
    # path directory of images in 'test_images' folder
    img_dir_path = 'test_images/'

    # path to 'test_image_1.png' image file
    file_num = 1
    img_file_path = img_dir_path + 'test_image_' + str(file_num) + '.png'
    
    # read image using opencv
    img = cv2.imread(img_file_path)
    
    print('\n============================================')
    print('\nFor test_image_' + str(file_num) + '.png')
    
    # detect shape properties from image
    detected_shapes = detect_shapes(img)
    print(detected_shapes)
    
    # display image with labeled shapes
    img = get_labeled_image(img, detected_shapes)
    cv2.imshow("labeled_image", img)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
    
    choice = input('\nDo you want to run your script on all test images ? => "y" or "n": ')
    
    if choice == 'y':

        for file_num in range(1, 16):
            
            # path to test image file
            img_file_path = img_dir_path + 'test_image_' + str(file_num) + '.png'
            
            # read image using opencv
            img = cv2.imread(img_file_path)
    
            print('\n============================================')
            print('\nFor test_image_' + str(file_num) + '.png')
            
            # detect shape properties from image
            detected_shapes = detect_shapes(img)
            print(detected_shapes)
            
            # display image with labeled shapes
            img = get_labeled_image(img, detected_shapes)
            cv2.imshow("labeled_image", img)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()


