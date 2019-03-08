import cv2
import numpy as np

def cursor_detection(img,img2_gray,img_cursor):

    imageGray = cv2.GaussianBlur(img2_gray,(3,3),0)
    templateGray = cv2.GaussianBlur(img_cursor,(3,3),0)

    templateGray = cv2.Canny(templateGray, 200, 600)
    imageGray = cv2.Canny(imageGray, 200, 600)

    #To resize the image
    scale_percent = 120 # percent of original size
    width = int(templateGray.shape[1] * scale_percent / 100)
    height = int(templateGray.shape[0] * scale_percent / 100)
    dim = (width, height)
    templateGray = cv2.resize(templateGray, dim, interpolation = cv2.INTER_AREA)

    # Find the matching template
    result = cv2.matchTemplate(imageGray, templateGray, cv2.TM_CCORR_NORMED)
    h, w = templateGray.shape

    threshold = 0.597
    loc = np.where(result >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (255, 0, 0), 2)

    # Show result
    cv2.imshow("Result", img)

    cv2.moveWindow("Template", 10, 50)
    cv2.moveWindow("Result", 150, 50)

    cv2.waitKey(0)

def find_cursor():
    #The Cursor_2.jpg is the template cursor.
    img_cursor = cv2.imread('Cursor_2.jpg', 0)
    #To change the input files, change the 'pos_' to the name of your image (without number) and determine the range in the for loop to loop over the images.
    for i in range(15):
        stri = 'pos_' + str((i+1)) + '.jpg'
        print(stri)
        img2_gray = cv2.imread(stri, 0)
        img = cv2.imread(stri)
        cursor_detection(img,img2_gray,img_cursor)

find_cursor()