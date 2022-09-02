# coding=utf-8
import cv2
import os
from newPutText import cv2_chinese_text

# 加载训练数据集文件
recogizer = cv2.face.LBPHFaceRecognizer_create()
recogizer.read('/Users/eddy_huang/Downloads/Python/facial recognition/Kpop-Facial-recognition/Kpop-Facial-Recognition/trainer/trainer.yml')
names = []


# 准备识别的图片
def face_detect_demo(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换为灰度
    face_detector = cv2.CascadeClassifier(

        r'/Users/eddy_huang/Downloads/Python/facial recognition/opencv-4.5.5/data/haarcascades/haarcascade_frontalface_alt2.xml')
    face = face_detector.detectMultiScale(gray, 1.1, 3, cv2.CASCADE_SCALE_IMAGE, (50, 100), (300, 300))
    for x, y, w, h in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=2)
        cv2.circle(img, center=(x + w // 2, y + h // 2), radius=w // 2, color=(0, 255, 0), thickness=1)
        # 人脸识别
        print(f"names:{names}")
        ids, confidence = recogizer.predict(gray[y:y + h, x:x + w])
        print(f"confidence:{confidence}")
        img = cv2_chinese_text(img, str(names[ids - 1]), (x + 10, y - 10), (0, 255, 0), 25)
    cv2.imshow('result', img)


def name():
    path = r'/Users/eddy_huang/Downloads/Python/facial recognition/Kpop-Facial-recognition/Kpop-Facial-Recognition/data'
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    for imagePath in imagePaths:
        name = str(os.path.split(imagePath)[1].split('.', 2)[1])
        names.append(name)


cap = cv2.VideoCapture('/Users/eddy_huang/Downloads/Python/facial recognition/Kpop-Facial-recognition/Kpop-Facial-Recognition/asset/773852032-1-208.mp4')

name()
while True:
    flag, frame = cap.read()
    if not flag:
        break
    face_detect_demo(frame)
    if ord(' ') == cv2.waitKey(5):
        break
cv2.destroyAllWindows()
cap.release()
print(names)
