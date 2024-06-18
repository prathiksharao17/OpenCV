#detecting red and purple fruits in an image 

import numpy as np
import cv2

img = cv2.imread(r"C:\Users\weelcome\Downloads\lde.png")

# Convert the image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# HSV range for red
redlower = np.array([0, 50, 100])
redupper = np.array([10, 150, 255])

# HSV range for purple
purplelower = np.array([125, 100, 100])
purpleupper = np.array([150, 255, 255])

# masks for red and purple 
redmask = cv2.inRange(hsv, redlower, redupper)
purplemask = cv2.inRange(hsv, purplelower, purpleupper)

# Dilate masks to make red and purple areas larger
kernel = np.ones((5, 5), np.uint8)
redmask_dilated = cv2.dilate(redmask, kernel, iterations=1)
purplemask_dilated = cv2.dilate(purplemask, kernel, iterations=1)

# contours for dilated masks
redcont, _ = cv2.findContours(redmask_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
purplecont, _ = cv2.findContours(purplemask_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

detect = img.copy()

# contours for red 
for contour in redcont:
    # Get the bounding box for each contour
    x, y, w, h = cv2.boundingRect(contour)
    # Draw the rectangle on the image
    cv2.rectangle(detect, (x, y), (x + w, y + h), (0, 255, 0), 2) 

# contours for purple
for contour in purplecont:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(detect, (x, y), (x + w, y + h), (255, 0, 255), 2)  

cv2.imshow('Original Image', img)
cv2.imshow('Red Mask', redmask)
cv2.imshow('Purple Mask', purplemask)
cv2.imshow('Dilated Red Mask', redmask_dilated)
cv2.imshow('Dilated Purple Mask', purplemask_dilated)
cv2.imshow('Detected Image', detect)

while True:
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cv2.destroyAllWindows()
