# -*- coding: utf-8 -*-
"""
Created on Sun May 24 02:09:00 2020

@author: erika
"""

import numpy as np
import cv2 as cv

def loadDataMask(size):
    
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    
    # getting train images #
    # first with mask #
    # this is ok #
    
    for i in range(1, 343):
        img = cv.imread('Train/Mask/' + str(i) + '.png', 1)
        img = cv.resize(img, (size, size))
        x_train.append(img)
        y_train.append(1)
        
    # for no mask #
        
    for i in range(15401, 15726):
        img = cv.imread('Train/No_Mask/' + str(i) + '.png', 1)
        img = cv.resize(img, (size, size))
        x_train.append(img)
        y_train.append(0)
        
        
    # Getting Test Images #
    # First with Mask #
    
    for i in range(1, 151):
        img = cv.imread('Validation/Mask/' +str(i) + '.png', 1)
        img = cv.resize(img, (size, size))
        x_test.append(img)
        y_test.append(1)
        
    # with no mask #
        
    for i in range(16000, 16200):
        img = cv.imread('Validation/No_Mask/' + str(i) + '.png', 1)
        img = cv.resize(img, (size, size))
        x_test.append(img)
        y_test.append(0)
        
    # Trasbformando em array #
        
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    
    return x_train, y_train, x_test, y_test  