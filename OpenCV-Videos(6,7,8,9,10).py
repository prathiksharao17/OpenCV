#video 6
import cv2
import numpy as np
img = cv2.imread(r"C:\Users\weelcome\Pictures\Saved Pictures\bookpage.jpg")
retval,threshold=cv2.threshold(img,12,255,cv2.THRESH_BINARY)
grayscaled=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
retval2,threshold2=cv2.threshold(grayscaled,12,255,cv2.THRESH_BINARY)
gaus=cv2.adaptiveThreshold(grayscaled,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
retval2,otsu=cv2.threshold(grayscaled,12,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('original',img)
cv2.imshow('threshold',threshold)
cv2.imshow('grayscaled',threshold2)
cv2.imshow('gauss',gaus)
cv2.imshow('otsu',otsu)
cv2.waitKey(0)
cv2.destroyAllWindows()


#video 7
import cv2
import numpy as np

cap=cv2.VideoCapture(0)

while True:
    _,frame=cap.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) #hsv=hue,saturation and value for range purposes
    
    lower_orange=np.array([20,30,90])
    upper_orange=np.array([180,255,180])
    mask=cv2.inRange(hsv,lower_orange,upper_orange)
    res=cv2.bitwise_and(frame,frame,mask=mask)
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k=cv2.waitKey(5) & 0xFF
    if k==27:
        break

cv2.destroyAllWindows()
cap.release()

#video 8
import cv2
import numpy as np

cap=cv2.VideoCapture(0)

while True:
    _,frame=cap.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) #hsv=hue,saturation and value for range purposes
    
    lower_orange=np.array([20,30,90])
    upper_orange=np.array([180,255,180])
    mask=cv2.inRange(hsv,lower_orange,upper_orange)
    res=cv2.bitwise_and(frame,frame,mask=mask)
    kernel=np.ones((15,15),np.float32)/225
    smoothed=cv2.filter2D(res,-1,kernel)
    blur=cv2.GaussianBlur(res,(15,15),0)
    median=cv2.medianBlur(res,15)
    bilat=cv2.bilateral(res,15,75,75)
    cv2.imshow('frame',frame)
    #cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    cv2.imshow('smoothed',smoothed)
    cv2.imshow('blur',blur)
    cv2.imshow('median',median)
    cv2.imshow('bilat',bilat)
    k=cv2.waitKey(5) & 0xFF
    if k==27:
        break

cv2.destroyAllWindows()
cap.release()


#video 9

import cv2
import numpy as np

cap=cv2.VideoCapture(0)

while True:
    _,frame=cap.read()
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) #hsv=hue,saturation and value for range purposes
    
    lower_orange=np.array([20,30,90])
    upper_orange=np.array([180,255,180])
    
    mask=cv2.inRange(hsv,lower_orange,upper_orange)
    res=cv2.bitwise_and(frame,frame,mask=mask)

    kernel=np.ones((5,5),np.uint8)
    erosion=cv2.erode(mask,kernel,iterations=1)
    dilation=cv2.dilate(mask,kernel,iterations=1)

    opening=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    closing=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)

    
    
    cv2.imshow('frame',frame)
    cv2.imshow('res',res)
    cv2.imshow('erosion',erosion)
    cv2.imshow('dilation',dilation)
    cv2.imshow('opening',opening)
    cv2.imshow('closing',closing)
    
    k=cv2.waitKey(5) & 0xFF
    if k==27:
        break

cv2.destroyAllWindows()
cap.release()


#video 10
import cv2
import numpy as np

cap=cv2.VideoCapture(0)

while True:
    
    _,frame=cap.read()
    laplacian=cv2.Laplacian(frame,cv2.CV_64F)
    sobelx=cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=5)
    sobely=cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=5)
    edges=cv2.Canny(frame,100,200)

    cv2.imshow('frame',frame)
    cv2.imshow('laplacian',laplacian)
    cv2.imshow('sobelx',sobelx)
    cv2.imshow('sobely',sobely)
    cv2.imshow('edges',edges)
    
    k=cv2.waitKey(5) & 0xFF
    if k==27:
        break

cv2.destroyAllWindows()
cap.release()




