import cv2, time
import os
from PIL import Image
cam=cv2.VideoCapture(0)
faceDetector = cv2.CascadeClassifier('haarcascade.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('latihwajah/training.xml')
a=0
while True:
    a=a+1
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceDetector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        frame = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if (id==1):
            id='Rafi'
        elif (id==2):
            id='Poppy'
        elif (id == 3):
            id = 'Ghulam'
        elif (id == 4):
            id = 'Niken'
        elif (id == 5):
            id = 'Yoel'
        cv2.putText(img,str(id),(x+40,y-10), cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0))
    cv2.imshow("Face Recognition", img)
    key = cv2.waitKey(1)
    if key == ord('x'):
        break
cam.release()
cv2.destroyAllWindows()