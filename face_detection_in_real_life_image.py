# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 20:49:53 2019

@author: Dharmik joshi
"""

import cv2

video = cv2.VideoCapture(0)

face_detection_data = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    
    check,img = video.read()

    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    #actual code
    
    faces = face_detection_data.detectMultiScale(gray_img,1.1,5)    
    
    for x,y,w,h in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),(4))        
        print(cv2.putText(img,"Face",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2))
    
    cv2.imshow("Face Detection",img)
    
    key = cv2.waitKey(1)
    
    if key == ord('e'):
        break
    

video.release()
cv2.destroyAllWindows()






