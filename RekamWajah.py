import cv2
wajahDir='datawajah'
cam=cv2.VideoCapture(0)
cam.set(3,640)
cam.set(6,480)
faceDetector=cv2.CascadeClassifier('haarcascade.xml')
id=input("Masukan Face ID : ")
print("Tatap wajah anda ke depan webcam. Tunggu proses pengambilan data selesai")
ambilData=0
while True:
    ret, img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetector.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        frame=cv2.rectangle(img, (x,y), (x+w, y+h),(0,255,0),2)
        namaFile = 'wajah.' + str(id) + '.' + str(ambilData) + '.jpg'
        cv2.imwrite(wajahDir + "/" + namaFile, frame)
        ambilData += 1
        roi_gray=gray[y:y+h,x:x+w]
        roi_img=img[y:y+h,x:x+w]

    cv2.imshow("Face Recognition", img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('x'):
        break
    elif ambilData > 30:
        break
print("pengambilan data selesai")
cam.release()
cv2.destroyAllWindows()