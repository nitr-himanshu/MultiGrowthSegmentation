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
sys.setrecursionlimit(100000) #for high growth
rcParams['figure.figsize'] = 50, 10

def convertToBlackAndWhite(img_path):
    '''
    converts image into binary (255,0)
    '''
    originalImage = cv2.imread(img_path)
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    (thresh, blackAndWhiteImage) = cv2.threshold(
        grayImage, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return blackAndWhiteImage

def showImg(img):
    '''
    Show an image in form of - and *
    '''
    for x in img:
        for y in x:
            if(y == 255):
                print("-",end="")
            elif(y == 0):
                print("*",end="")
        print()
        
def showImgLabel(img):
    '''
    show an image label positions
    '''
    for x in img:
        for y in x:
            if(y == 0):
                print("-",end="")
            else:
                print(y,end="")
        print()

def update(i,j,label,img,label_img):
    '''
    Update label of given position (i,j) and its neighbours recursively.
    If growth is high and recursion depth is more hence need to increase 
    recursion depth other wise error.
    '''
    row = label_img.shape[0]
    col = label_img.shape[1]
    
    label_img[i][j] = label #setting given position label
    
    neighbours = list()
    
    # all 8 neighbours coordinates
    coordinates = [(i-1,j-1),(i-1,j),(i-1,j+1),
                   (i,j-1),(i,j+1),
                   (i+1,j-1),(i+1,j),(i+1,j+1)]
    
    # checks for valid coordinates, black unlabelled pixels
    for x,y in coordinates:
        if(x >=0 and x < row and y >= 0 and y < col and 
           img[x][y] == 0 and label_img[x][y] == 0):
            neighbours.append((x,y))
    
    for x,y in neighbours:
        update(x,y,label,img,label_img)

def label(img):
    '''
    Return a matrix of same dimension of image. 
    Consists label of corresponding pixel. If pixel white then label = 0
    '''
    label_img = np.zeros(img.shape,dtype="int")
    
    row = label_img.shape[0]
    col = label_img.shape[1]
    
    # counts no of label (segmentation)
    count = 0
    
    for i in range(row):
        for j in range(col):
            # check for unlabelled black pixel
            if(img[i][j] == 0 and label_img[i][j] == 0):
                count += 1
                update(i,j,count,img,label_img)   
            
    return [count, label_img]

def boxing(img,label_img):
    '''
    Return a boxed image based on labels
    '''
    row = label_img.shape[0]
    col = label_img.shape[1]
    
    # class for defaultdict to store boundary coordinate of a label
    class c:
        xmin = +np.inf
        ymin = +np.inf
        xmax = -np.inf
        ymax = -np.inf
    
    
    label_boundary = defaultdict(c)
    
    # new image with boundary box drawn
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
    
    # drawing box                
    for l in label_boundary.keys():
        obj = label_boundary[l]
        
        # vertical lines
        for i in range(obj.xmin,obj.xmax+1):
            boxedImg[i][obj.ymin] = 80
            boxedImg[i][obj.ymax] = 80
        
        # horizontal lines
        for j in range(obj.ymin,obj.ymax+1):
            boxedImg[obj.xmin][j] = 80
            boxedImg[obj.xmax][j] = 80
            
    return [boxedImg, label_boundary]

def segmentation(img,label_boundary,fol_path = "",file_name = "segment_"):
    '''
    Segment image according to labels. Return count of segment.
    save the segmented image.
    '''
    c = 1
    for l in label_boundary.keys():
        cropped_img = img[label_boundary[l].xmin:label_boundary[l].xmax+1,
                          label_boundary[l].ymin:label_boundary[l].ymax+1]
        cv2.imwrite(fol_path + file_name + str(c)+".png",cropped_img)
        c+=1
    return c

def growth(img):
    '''
    Grow the image by one pixel. a black pixel will make
    all its neighbour black.
    '''
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
                
                # valid white pixels
                for x,y in coordinates:
                    if(x >=0 and x < row and y >=0 and y < col 
                       and img[x][y] != 0):
                        white_neighbours.append((x,y))
                
                # convert into black pixel
                for x,y in white_neighbours:
                    grownImg[x][y] = 0
    return grownImg
    
def growthN(img,n):
    '''
    Grow an image for n times.
    '''
    grownImg = img
    for i in range(n):
        grownImg = growth(grownImg)
    return grownImg
    


if __name__ == "__main__":
    
    img = convertToBlackAndWhite("d1.png")
    count , label_img = label(img)
    boxedImg, label_boundary = boxing(img,label_img)
    segmentation(img,label_boundary,"./segment/","d1_")
    
#    plt.subplot(3,1,1)
#    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))