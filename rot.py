import math
import numpy as np
import cv2
from matplotlib import pyplot as plt


import imutils


def distance(x1,x2,y1,y2):
    d=math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
    return d


path="C:/Image_Aesthetic/duckblue.png"

img = cv2.imread(path)

plt.imshow(img)

mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)

fgdModel = np.zeros((1,65),np.float64)

rect = (50,20,250,150) 

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

mask3 = np.where(~((mask==2)|(mask==0)),0,1).astype('uint8')



img1 = img*mask2[:,:,np.newaxis]

plt.imshow(img1)

#--------CENTERS OF Grid------------------


print("Size of image= ", img1.shape)

x1=img1.shape[1]/3
y1=img1.shape[0]/3
print("GridC1 = (",x1,", ",y1,")")
x2=2*img1.shape[1]/3
y2=img1.shape[0]/3
print("GridC2 = (",x2,",",y2,")")
x3=img1.shape[1]/3
y3=2*img1.shape[0]/3
print("GridC3 = (",x3,",",y3,")")
x4=2*img1.shape[1]/3
y4=2*img1.shape[0]/3
print("GridC4 = (",x4,",",y4,")")


#--------CENTROID------------------

gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)[1]

# find contours in the thresholded image
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
# loop over the contours
for c in cnts:
	# compute the center of the contour
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY= int(M["m01"] / M["m00"])


# draw the contour and center of the shape on the image
cv2.drawContours(img1, [c], -1, (0, 255, 0), 1)
cv2.circle(img1, (cX, cY), 5, (0, 255, 0), -1)   
cv2.putText(img1, "center", (cX - 50, cY - 20),
cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
   
print("Centroid of Object= (",cX,",",cY,")")


d1=distance(cX,x1,cY,y1) 
d2=distance(cX,x2,cY,y2) 
d3=distance(cX,x3,cY,y3) 
d4=distance(cX,x4,cY,y4) 

if min(d1,d2,d3,d4)<=30:                    #THRESHOLD DISTANCE
    print("Follows ROT")
    print("Appealing")
else:
    print("Does not follow ROT")
    print("Not Appealing")

cv2.line(img1, (int(img1.shape[1]/3), 0), (int(img1.shape[1]/3), img1.shape[0]), (255, 0, 0), 2)
cv2.line(img1, (int(2*img1.shape[1]/3), 0), (int(2*img1.shape[1]/3), img1.shape[0]), (255, 0, 0), 2)

cv2.line(img1, (0,int(img1.shape[0]/3)), (img1.shape[1],int(img1.shape[0]/3)), (255, 0,0), 2)
cv2.line(img1, (0,int(2*img1.shape[0]/3)), (img1.shape[1],int(2*img1.shape[0]/3)), (255, 0,0), 2)

plt.imshow(img1)
plt.title("Foreground Image")
plt.colorbar()
plt.show()



