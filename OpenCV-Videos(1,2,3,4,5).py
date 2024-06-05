#Video 1
import cv2
import numpy as np
import matplotlib.pyplot as plt

#show image using cv
img=cv2.imread(r"C:\Users\weelcome\Pictures\Saved Pictures\watch.jpg",cv2.IMREAD_GRAYSCALE)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#show image using matplotlib
plt.imshow(img,cmap='gray',interpolation='bicubic') 
plt.plot([50,100],[80,100],'c',linewidth=5) #to plot on the image 
plt.show()

#to save an image 
cv2.imwrite('watchgray.png',img) 


#Video 2 - Interacting with a webcam or a video file
import cv2
import numpy as np

cap=cv2.VideoCapture(0)
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))
while True:
    ret, frame = cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',frame)
    cv2.imshow('gray',gray) 
    if cv2.waitKey(1) & 0xFF==ord('q'): #captures video until q key is pressed
        break
cap.release()
out.release()
cv2.destroyAllWindows()

#Video 3
import cv2
import numpy as np
img = cv2.imread(r"C:\Users\weelcome\Pictures\Saved Pictures\watch.jpg",cv2.IMREAD_COLOR)

#for color of the line - its BGR format 
#blue (255,0,0) green (0,255,0) red (0,0,255)
#white(255,255,255) black (0,0,0)

cv2.line(img , (0,0),(150,150),(255,255,255),15) 
cv2.rectangle(img, (15,25),(200,150),(0,255,0),5)
cv2.circle(img, (100,63),55,(0,0,255),-1)
pts=np.array([[10,5],[20,30],[70,20],[50,10]],np.int32) 
#pts=pts.reshape((-1,1,2))
cv2.polylines(img,[pts],True,(0,255,255),5)
#true is because we want to connect the last point to the first point

font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,'LEARNING OpenCV!!!',(0,130),font,1,(200,255,255),2,cv2.LINE_AA)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Video 4
#The image is read from the specified file.
#A single pixel is modified to white.
#A region of the image is set to white.
#A portion of the image is copied and pasted into another location within the same image.
#The modified image is displayed until a key is pressed.

import cv2
import numpy as np
img = cv2.imread(r"C:\Users\weelcome\Pictures\Saved Pictures\watch.jpg",cv2.IMREAD_COLOR)


img[55,55]=[255,255,255]
px=img[55,55]
print(px)

#region of image
img[100:150,100:150]=[255,255,255]

watch_face=img[37:111,107:194]
img[0:74,0:87]=watch_face #HAS TO BE THE SAME SIZE.(The Difference)



cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Video 4
#The image is read from the specified file.
#A single pixel is modified to white.
#A region of the image is set to white.
#A portion of the image is copied and pasted into another location within the same image.
#The modified image is displayed until a key is pressed.

import cv2
import numpy as np
img = cv2.imread(r"C:\Users\weelcome\Pictures\Saved Pictures\watch.jpg",cv2.IMREAD_COLOR)


img[55,55]=[255,255,255]
px=img[55,55]
print(px)

#region of image
img[100:150,100:150]=[255,255,255]

watch_face=img[37:111,107:194]
img[0:74,0:87]=watch_face #HAS TO BE THE SAME SIZE.(The Difference)



cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#video 5
import cv2
import numpy as np

img1=cv2.imread(r"C:\Users\weelcome\Pictures\Saved Pictures\3D-Matplotlib.png")
img2=cv2.imread(r"C:\Users\weelcome\Pictures\Saved Pictures\mainsvmimage.png")
img3=cv2.imread(r"C:\Users\weelcome\Pictures\Saved Pictures\mainlogo.png")

#add=img1+img2 #just adds the two images
#add1=cv2.add(img1,img2) #adds the pixel values of the two images 
#weighted=cv2.addWeighted(img1,0.6,img2,0.4,0) #superimpose images on each other 

rows,cols,channels=img3.shape
roi=img1[0:rows,0:cols]

img2gray=cv2.cvtColor(img3,cv2.COLOR_BGR2GRAY)
ret,mask=cv2.threshold(img2gray,220,255, cv2.THRESH_BINARY_INV) 

mask_inv=cv2.bitwise_not(mask)

img1_bg=cv2.bitwise_and(roi,roi,mask=mask_inv)
img3_fg=cv2.bitwise_and(img3,img3,mask=mask)

dst=cv2.add(img1_bg,img3_fg)
img1[0:rows,0:cols]=dst

cv2.imshow('res',img1)

 
#cv2.imshow('add',add)
#cv2.imshow('add1',add1)
#cv2.imshow('addWeighted',weighted)
cv2.imshow('mask',mask)

cv2.waitKey(0)
cv2.destroyAllWindows()



