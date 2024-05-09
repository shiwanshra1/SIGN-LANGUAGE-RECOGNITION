import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as qw
import math
cap = cv2.VideoCapture(0)
dect = HandDetector(maxHands=1)
classifier = Classifier("model/keras_model.h5","model/labels.txt")
set = 20
imgsize = 300
counter = 0

folder = "data/hello"
labels = ["A","B","C","D","E","F","G","H","I","J","K","L","HELLO"]
while True :
    sucess , img = cap.read()
    imgoutput = img.copy()
    hands , img = dect.findHands(img)
    if hands:
        hands = hands[0]
        x,y,w,h = hands['bbox']
        imgwite = qw.ones((imgsize,imgsize,3),qw.uint8)*255
        imgcrop = img[y-set:y+h+set, x-set:x+w+set]
        aspectratio = h/w
        if aspectratio>1:
            k = imgsize/h
            wcal = math.ceil(k*w)
            imgresize = cv2.resize(imgcrop, (wcal,imgsize))
            imgresizeshape = imgresize.shape
            wgap = math.ceil((imgsize-wcal)/2)
            imgwite[ : , wgap:wcal+wgap] = imgresize
            prediction , index =classifier.getPrediction(imgwite)
            print(prediction,index)
        else:
            k = imgsize/w
            hcal = math.ceil(k*h)
            imgresize = cv2.resize(imgcrop, (imgsize,hcal))
            imgresizeshape = imgresize.shape
            hgap = math.ceil((imgsize-hcal)/2)
            imgwite[hgap:hcal+hgap, : ] = imgresize
            prediction , index =classifier.getPrediction(imgwite)


        cv2.putText(imgoutput,labels[index],(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
        # cv2.rectangle(imgoutput,(x-set,y-set),(x+w+set,y+h+set),(255,0,255),3)
        cv2.imshow("imagecrop",imgcrop)
        cv2.imshow("imagewhite",imgwite)
    cv2.imshow("image",imgoutput)
    cv2.waitKey(1)