# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 10:14:42 2019

@author: Anne
"""

import numpy as np

import cv2

from matplotlib import pyplot as plt

path="C:/Image_Aesthetic/DOF/dof1.jpg"

img = cv2.imread(path)

cv2.imshow('image',img)

#cv2.waitKey(0)

mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)

fgdModel = np.zeros((1,65),np.float64)

rect = (200,200,400,400) 

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

mask3 = np.where(~((mask==2)|(mask==0)),0,1).astype('uint8')

img1 = img*mask2[:,:,np.newaxis]
img2 = img*mask3[:,:,np.newaxis]

plt.imshow(img)
plt.title("Original Image")
plt.colorbar()
plt.show()

plt.imshow(img1)
plt.title("Foreground Image")
plt.colorbar()
plt.show()

plt.imshow(img2)
plt.title("Background Image")
plt.colorbar()
plt.show()

