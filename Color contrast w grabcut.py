#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 17:59:13 2019

@author: anuja
"""

import cv2
from matplotlib import pyplot as plt

# read original image, in full color, based on command
# line argument
path1="C:/Image_Aesthetic/bbb.png"
path2="C:/Image_Aesthetic/ccc.png"

plt.title("Foreground")

ForegroundImg = cv2.imread(path2)
#cv2.imshow("Image", ForegroundImg)
plt.imshow(ForegroundImg)
plt.show()


plt.title("Background")

#cv2.waitKey(0)
BackgroundImg = cv2.imread(path1)
#cv2.imshow("Image", BackgroundImg)
plt.imshow(BackgroundImg)
plt.show()


# split into channels
channels = cv2.split(ForegroundImg)
channels1 = cv2.split(BackgroundImg)

colors = ("b", "g", "r")
colors1 = ("b", "g", "r")

Red = channels [0]
Green = channels[1]
Blue = channels[2]

Red1 = channels1 [0]
Green1 = channels1[1]
Blue1 = channels1[2]

Rmean=sum(Red) /len(Red)
Gmean=sum(Green) /len(Green)
Bmean=sum(Blue) /len(Blue)

Rmean1=sum(Red1) /len(Red1)
Gmean1=sum(Green1) /len(Green1)
Bmean1=sum(Blue1) /len(Blue1)

Avg=(Rmean+Bmean+Gmean)/3
color_contrast=max(Avg)
print("Foreground Color contrast: ",color_contrast)

Avg1=(Rmean1+Bmean1+Gmean1)/3
color_contrast1=max(Avg1)
print("Background Color contrast: ",color_contrast1)

newcolor=abs(color_contrast1-color_contrast)
print("Difference in Color contrast: ",newcolor)


threshold=0.1
if newcolor < threshold:                    #THRESHOLD
   text = "Low Contrast"
else:
    text = "High Contrast"
    
print(text)    

plt.title("Foreground")
plt.xlabel("Pixels")
plt.ylabel("Mean Color value")
plt.plot(Rmean,color='Red')
plt.plot(Gmean,color='Green')
plt.plot(Bmean,color='Blue')
plt.show()

plt.title("Background")
plt.xlabel("Pixels")
plt.ylabel("Mean Color value")
plt.plot(Rmean1, color='Red')
plt.plot(Gmean1,color='Green')
plt.plot(Bmean1,color='Blue')

plt.show()

# list to select colors of each channel line
print(colors)
print(colors1)
# create the histogram plot, with three lines, one for
# each color
plt.xlim([0, 256])
plt.ylim([0,300])
for(channel, c) in zip(channels, colors):
    histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plt.plot(histogram, color = c)

plt.xlabel("Pixels")
plt.ylabel("Channel Color value")
plt.title("Foreground")
plt.show()

plt.xlim([0, 256])
plt.ylim([0,800])

for(channel, c) in zip(channels1, colors1):
    histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plt.plot(histogram, color = c)

plt.xlabel("Pixels")
plt.ylabel("Channel Color value")
plt.title("Background")
plt.show()

cv2.waitKey(0)

cv2.waitKey(0)
cv2.destroyAllWindows()
