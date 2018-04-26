import face_keras as face
import sys, os
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import time

cascade_path = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)
cam = cv2.VideoCapture(-1)
color = (255, 255, 255)

image_size = 32
categories = ["Angelina_Jolie", "Bill_Clinton", "Jiang_Zemin", "Jose_Maria_Aznar", "Naomi_Watts", "Rudolph_Giuliani","Erwin_Wu"]
key=True

while(key):
        ret, frame = cam.read()
        
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	facerect = cascade.detectMultiScale(gray,1.3,5)
        img = frame
        for rect in facerect:
            x = rect[0]
            y = rect[1]
            w = rect[2]
            h = rect[3]
	    img = img[y:y+h, x:x+w]
	    height, width = img.shape[:2]
	    if (height>0 and width>0):
		X = []
		img = cv2.resize(img,(image_size,image_size))
		X.append(img)
		X = np.array(X)
		X  = X.astype("float")  / 256
		model = face.build_model(X.shape[1:])
		model.load_weights("./image/face-model.h5")
		pre = model.predict(X)
		for i,item in enumerate(pre[0]):
			if item>0.9:
				print categories[i]
				print pre
		cv2.imshow("Show FRAME Image", img)

        k = cv2.waitKey(1000)

        if k == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()

