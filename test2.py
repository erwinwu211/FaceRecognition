import face_keras as face
import sys, os
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import time

out_dir = "./image/Erwin_Wu"
cascade_path = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)
cap = cv2.VideoCapture(-1)
color = (255, 255, 255)

image_size = 32
categories = ["Angelina_Jolie", "Bill_Clinton", "Jiang_Zemin", "Jose_Maria_Aznar", "Naomi_Watts", "Rudolph_Giuliani"]
i=0
while(True):
	ret,frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	facerect = cascade.detectMultiScale(gray,1.3,5)
	for rect in facerect:
		x = rect[0]
	    	y = rect[1]
	    	w = rect[2]
	    	h = rect[3]
	    	img = frame[y:y+h, x:x+w]
		img = cv2.resize(img,(image_size,image_size))
		cv2.imshow('frame',img)
		cv2.imwrite(out_dir+"/frame"+str(i)+".jpg",img)
		i=i+1
	if cv2.waitKey(30)== ord('q'):
		break;

cap.release()
cv2.destroyAllWindows()
