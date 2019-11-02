# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 17:32:38 2019

@author: Anne
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt


def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()


path="C:/Image_Aesthetic/3blue.png"

img = cv2.imread(path)

#cv2.imshow('image',img
plt.imshow(img)

#cv2.waitKey(0)

mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)

fgdModel = np.zeros((1,65),np.float64)
#from left, from top, from right, from bottom
rect = (100,75,200,150) 

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

mask3 = np.where(~((mask==2)|(mask==0)),0,1).astype('uint8')



img1 = img*mask3[:,:,np.newaxis]

plt.imshow(img1)


gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
fm = variance_of_laplacian(gray)
text = "Not Blurry"
 
	# if the focus measure is less than the supplied threshold,
	# then the image should be considered "blurry"
    
print(fm)
if fm < 650:                            #THRESHOLD
	text = "Blurry\nAppealing"
else:
   text="Not Blurry\nNot Appealing"
print(text)
 
	# show the image
#cv2.putText(img1, "{}: {:.2f}".format(text, fm), (5, 30),
#	cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
#cv2.imshow("Image", img1)
key = cv2.waitKey(0)

