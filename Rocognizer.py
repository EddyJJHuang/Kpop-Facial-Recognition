import os
import cv2
import sys
from PIL import Image
import numpy as np

recogizer=cv2.face.LBPHFaceRecognizer_create()
recogizer.read('/Users/eddy_huang/Downloads/Python/facial recognition/Kpop-Facial-recognition/Kpop-Facial-Recognition/trainer/trainer.yml')
names=[]

def face_detect_demo(img):
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#转换为灰度
    face_detector=cv2.CascadeClassifier('/Users/eddy_huang/Downloads/Python/facial recognition/opencv-4.5.5/data/haarcascades/haarcascade_frontalface_alt2.xml')
    face=face_detector.detectMultiScale(gray,1.1,5,cv2.CASCADE_SCALE_IMAGE,(70,70),(300,300))
    #face=face_detector.detectMultiScale(gray)
    for x,y,w,h in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),color=(0,0,255),thickness=5)
        cv2.circle(img,center=(x+w//2,y+h//2),radius=w//2,color=(255,255,0),thickness=1)
        # 人脸识别
        ids, confidence = recogizer.predict(gray[y:y + h, x:x + w])
        #print('标签id:',ids,'置信评分：', confidence)
        cv2.putText(img,str(names[ids-1]), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
    cv2.imshow('result',img)

def name():
    path = '/Users/eddy_huang/Downloads/Python/facial recognition/Kpop-Facial-recognition/Kpop-Facial-Recognition/data'
    #names = []
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    for imagePath in imagePaths:
       name = str(os.path.split(imagePath)[1].split('.',2)[1])
       names.append(name)

cap=cv2.VideoCapture('/Users/eddy_huang/Downloads/Python/facial recognition/Kpop-Facial-recognition/Kpop-Facial-Recognition/asset/test video.mp4')
name()
while True:
    flag,frame=cap.read()
    if not flag:
        break
    face_detect_demo(frame)
    if ord(' ') == cv2.waitKey(10):
        break
cv2.destroyAllWindows()
cap.release()