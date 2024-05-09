import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as qw
import math
import time
cap = cv2.VideoCapture(0)
dect = HandDetector(maxHands=2)
set = 20
imgsize = 300
counter = 0

folder = "data/hello"

while True :
    sucess , img = cap.read()
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
        else:
            k = imgsize/w
            hcal = math.ceil(k*h)
            imgresize = cv2.resize(imgcrop, (imgsize,hcal))
            imgresizeshape = imgresize.shape
            hgap = math.ceil((imgsize-hcal)/2)
            imgwite[hgap:hcal+hgap, : ] = imgresize
        cv2.imshow("imagecrop",imgcrop)
        cv2.imshow("imagewhite",imgwite)
    cv2.imshow("image",img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter+=1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgwite)
        print(counter)