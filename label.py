#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 23:21:46 2020

@author: shield
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def convertToBlackAndWhite(img_path):
    originalImage = cv2.imread(img_path)
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(
        grayImage, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return blackAndWhiteImage

def showImg(img):
    for x in img:
        for y in x:
            print(str(y).zfill(3),end="")
        print()

def showImg2(img):
    for x in img:
        for y in x:
            if(y == 255):
                print("-",end="")
            elif(y == 0):
                print("*",end="")
        print()
        
def showImg3(img):
    for x in img:
        for y in x:
            if(y == 0):
                print("-",end="")
            else:
                print(y,end="")
        print()

def update(i,j,label,img,label_img):
    
    row = label_img.shape[0]
    col = label_img.shape[1]
    
    label_img[i][j] = label
    neighbours = list()
    coordinates = [(i-1,j-1),(i-1,j),(i-1,j+1),
                   (i,j-1),(i,j+1),
                   (i+1,j-1),(i+1,j),(i+1,j+1)]
    for x,y in coordinates:
        if(x >=0 and x < row and y >=0 and y < col 
           and img[x][y] == 0 and label_img[x][y] == 0):
            neighbours.append((x,y))
    
    for x,y in neighbours:
        update(x,y,label,img,label_img)


def label(img):
    
    label_img = np.zeros(img.shape,dtype="int")
    row = label_img.shape[0]
    col = label_img.shape[1]
    count = 0
    
    for i in range(row):
        for j in range(col):
            if(img[i][j] == 0 and label_img[i][j] == 0):
                count += 1
                update(i,j,count,img,label_img)   
                
    return label_img

if __name__ == "__main__":
    img = convertToBlackAndWhite("2.jpg")
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    label_img = label(img)
    showImg3(label_img)