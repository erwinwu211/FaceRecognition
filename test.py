import cv2
import numpy as np
import os,glob
from sklearn import cross_validation
from keras.preprocessing.image import load_img, img_to_array

root_dir = "./Dataset/"
out_dir = "./Out/"
categories = ["Angelina_Jolie", "Bill_Clinton", "Jiang_Zemin", "Jose_Maria_Aznar", "Naomi_Watts", "Rudolph_Giuliani"]
nb_classes = len(categories)
image_size = 32 


cascade_path = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)
X = []
Y = []

for idx, cat in enumerate(categories):
	files = glob.glob(root_dir + "/" + cat + "/*")
	print("---", cat, "dealing")
	os.mkdir(out_dir+cat)
	for i, f in enumerate(files):
		img = cv2.imread(f)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		facerect = cascade.detectMultiScale(gray,1.3,5)
		for rect in facerect:
			x = rect[0];y = rect[1];w = rect[2];h = rect[3];
	    	img = img[y:y+h, x:x+w]
		img = cv2.resize(img,(image_size,image_size))
		cv2.imshow('frame',img)
		cv2.imwrite(out_dir+"/"+cat+"/frame"+str(i)+".jpg",img)
		cv2.waitKey(30)

