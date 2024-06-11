#COLOR DETECTION USING CSV DATASET and OPENCV

import cv2 as cv
import pandas as pd

# Read the image
img = cv.imread(r"C:\Users\weelcome\Pictures\Saved Pictures\opencv-proj-image.jpg")

# declare global variables
click = False
r = g = b = xpos = ypos = 0

# Read the CSV file containing the color data
index = ['color', 'color_name', 'hexadecimal', 'R', 'G', 'B']
csv = pd.read_csv(r"C:\Users\weelcome\Downloads\colors.csv", names=index, header=None)

# Function to get the coordinates of the mouse click
def coordinates(event, x, y, flags, param):
    global b, g, r, xpos, ypos, click
    if event == cv.EVENT_LBUTTONDBLCLK:
        click = True
        xpos = x
        ypos = y
        b, g, r = img[y, x]
        b = int(b)
        g = int(g)
        r = int(r)

# Function to get the color name from the dataset
def get_color(R, G, B):
    min_dist = 10000
    cname = ""
    for i in range(len(csv)):
        d = abs(R - int(csv.loc[i, 'R'])) + abs(G - int(csv.loc[i, 'G'])) + abs(B - int(csv.loc[i, 'B']))
        if d < min_dist:
            min_dist = d
            cname = csv.loc[i, 'color_name']
    return cname

# Setting up the mouse callback function
cv.namedWindow('image')
cv.setMouseCallback('image', coordinates)

while True:
    cv.imshow('image', img)
    if click:
        # Draw a rectangle and display color name and RGB values
        cv.rectangle(img, (20, 20), (790, 80), (b, g, r), -1)
        text = get_color(r, g, b) + ' R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)
        
        # Adjust text color based on the background brightness
        text_color = (255, 255, 255) if r + g + b < 600 else (0, 0, 0)
        
        # Draw the text
        cv.putText(img, text, (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv.LINE_AA)
        
        click = False

    # Break the loop when 'Esc' key is pressed
    if cv.waitKey(20) & 0xFF == 27:
        break

cv.destroyAllWindows()

