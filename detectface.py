import os 
import cv2

cascPath = 'facedetect/haarcascade_frontalface_default'
faceCascade = cv2.CascadeClassifier(cascPath)
img = cv2.imread("/Users/macduffolusa/documents/learn/ml/facedetect/imagetest.png")
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
cv2.imshow("Faces found", img)
cv2.waitKey()
