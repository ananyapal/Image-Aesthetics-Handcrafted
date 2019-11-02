# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 10:14:42 2019

@author: Anne
"""

import numpy as np

import cv2

from matplotlib import pyplot as plt

path="C:/Users/Anne/Desktop/High Colour Contrast Images/93.jpg"

img = cv2.imread(path)

cv2.imshow('image',img)

#cv2.waitKey(0)

mask = np.zeros(img.shape[:2],np.uint8)

bgdModel = np.zeros((1,65),np.float64)

fgdModel = np.zeros((1,65),np.float64)

rect = (1,20,450,800) 

cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

img = img*mask2[:,:,np.newaxis]



plt.imshow(img)

plt.colorbar()

plt.show()