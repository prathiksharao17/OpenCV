#video 11
import cv2
import numpy as np

img_bgr=cv2.imread(r"C:\Users\weelcome\Pictures\Saved Pictures\opencv-template-matching-python-tutorial.jpg")
img_gray=cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)

template=cv2.imread(r"C:\Users\weelcome\Pictures\Saved Pictures\opencv-template-for-matching.jpg",0)
w,h=template.shape[::-1]

res=cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold=0.7
loc=np.where(res>=threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_bgr, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

cv2.imshow('Detected at thresh ',img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()


#video 12
import cv2
import numpy as np
import matplotlib.pyplot as plt

img=cv2.imread(r"C:\Users\weelcome\Pictures\Saved Pictures\opencv-python-foreground-extraction-tutorial.jpg")
mask = np.zeros(img.shape[:2],np.uint8)
bgdModel = np.zeros((1,65),np.float64)
fgdModel = np.zeros((1,65),np.float64)
rect = (161,79,150,150)

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
img = img*mask2[:,:,np.newaxis]

plt.imshow(img)
plt.colorbar()
plt.show()


#Video 13
import numpy as np
import cv2

img = cv2.imread(r"C:\Users\weelcome\Pictures\Saved Pictures\opencv-corner-detection-sample.jpg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int0(corners)
for corner in corners:
    x,y = corner.ravel()
    cv2.circle(img,(x,y),3,255,-1)
    
cv2.imshow('Corner',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#video 14
import cv2
import numpy as np
import matplotlib.pyplot as plt

img1=cv2.imread(r"C:\Users\weelcome\Pictures\Saved Pictures\opencv-feature-matching-template.jpg",0)
img2=cv2.imread(r"C:\Users\weelcome\Pictures\Saved Pictures\opencv-feature-matching-image.jpg",0)

orb=cv2.ORB_create()

kp1,des1=orb.detectAndCompute(img1,None)
kp2,des2=orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
plt.imshow(img3)
plt.show()


#Video 15
import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(r"C:\Users\weelcome\Videos\people-walking.mp4")
fgbg = cv2.createBackgroundSubtractorMOG2()

while(1):
    ret, frame = cap.read()

    fgmask = fgbg.apply(frame)
 
    cv2.imshow('fgmask',frame)
    cv2.imshow('frame',fgmask)

    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    

cap.release()
cv2.destroyAllWindows()




