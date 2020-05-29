# -*- coding: utf-8 -*-
"""
Created on Mon May 25 03:41:02 2020

@author: erika
"""


import numpy as np
import keras
from keras.models import load_model
from keras.optimizers import adam
import cv2 as cv
import matplotlib.pyplot as plt

# Função só rpa facilitar #

def auxiliar(value):
    if(value == 0):
        return "Mask not detected"
    else:
        return "Mask detected"

# Importando o modelo #
        
size = 32

model = load_model('testando')

model.compile(adam(lr=0.0001), loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Importando o detector de face #

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Importando a img #

img = cv.imread('5.jpg', 1)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Frontal Face #

faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    # (x, y) é o ponto de cima #
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cropped = img[y:y+h, x:x+w]
    cropped = cv.resize(cropped, (size, size))
    cropped = np.reshape(cropped, (1, size, size, 3))
    classes = model.predict_classes(cropped)
    cv.putText(img, auxiliar(classes), (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0 ,0), 2)
 

cv.imshow('img', img)
cv.waitKey(0)
cv.destroyAllWindows()