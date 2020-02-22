#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 23:21:46 2020

@author: shield
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
from pylab import rcParams
import sys
sys.setrecursionlimit(100000)
rcParams['figure.figsize'] = 50, 10

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
#    print(id(label_img))
    label_img[i][j] = label
    neighbours = list()
    coordinates = [(i-1,j-1),(i-1,j),(i-1,j+1),
                   (i,j-1),(i,j+1),
                   (i+1,j-1),(i+1,j),(i+1,j+1)]
    for x,y in coordinates:
        if(x >=0 and x < row and y >= 0 and y < col and img[x][y] == 0 and label_img[x][y] == 0):
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

def boxing(img,label_img):
    
    row = label_img.shape[0]
    col = label_img.shape[1]
    
    class c:
        xmin = +np.inf
        ymin = +np.inf
        xmax = -np.inf
        ymax = -np.inf
    
    
    label_boundary = defaultdict(c)
    boxedImg = np.array(img,copy=True)
    
    for i in range(row):
        for j in range(col):
            if(label_img[i][j] != 0):
                label = label_img[i][j]
                
                if(i < label_boundary[label].xmin):
                    label_boundary[label].xmin = i
                    
                if(i > label_boundary[label].xmax):
                    label_boundary[label].xmax = i
                    
                if(j < label_boundary[label].ymin):
                    label_boundary[label].ymin = j
                    
                if(j > label_boundary[label].ymax):
                    label_boundary[label].ymax = j
                    
    for l in label_boundary.keys():
        obj = label_boundary[l]
        
        for i in range(obj.xmin,obj.xmax+1):
            boxedImg[i][obj.ymin] = 80
            boxedImg[i][obj.ymax] = 80
            
        for j in range(obj.ymin,obj.ymax+1):
            boxedImg[obj.xmin][j] = 80
            boxedImg[obj.xmax][j] = 80
            
    return [boxedImg, label_boundary]

def segmentation(img,label_boundary):
    
    c = 1
    for l in label_boundary.keys():
        cropped_img = img[label_boundary[l].xmin:label_boundary[l].xmax+1,
                          label_boundary[l].ymin:label_boundary[l].ymax+1]
        cv2.imwrite("segment_"+str(c)+".png",cropped_img)
        c+=1
    return c

def growth(img):
    
    grownImg = np.array(img,copy=True)
    row = img.shape[0]
    col = img.shape[1]
    
    for i in range(row):
        for j in range(col):
            if(img[i][j] == 0):
                white_neighbours = list()
                coordinates = [(i-1,j-1),(i-1,j),(i-1,j+1),
                               (i,j-1),(i,j+1),
                               (i+1,j-1),(i+1,j),(i+1,j+1)]
                for x,y in coordinates:
                    if(x >=0 and x < row and y >=0 and y < col 
                       and img[x][y] != 0):
                        white_neighbours.append((x,y))
                
                for x,y in white_neighbours:
                    grownImg[x][y] = 0
    return grownImg
    
def growthN(img,n):
    grownImg = img
    for i in range(n):
        grownImg = growth(grownImg)
    return grownImg
    


if __name__ == "__main__":
    img = convertToBlackAndWhite("pq.jpg")
    plt.subplot(3,1,1)
    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
#    img2 = growthN(img,12)
#    plt.subplot(3,1,2)
#    plt.imshow(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB))
#    cv2.imwrite("pq.jpg",img2)
    label_img = label(img)
#    showImg3(label_img)
    boxedImg, label_boundary = boxing(img,label_img)
#    segmentation(img,label_boundary)
    plt.subplot(3,1,3)
    plt.imshow(cv2.cvtColor(boxedImg,cv2.COLOR_BGR2RGB))