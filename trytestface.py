import cv2
import numpy as np
import face_recognition

imgChris = face_recognition.load_image_file("ImagesBasic/chris.jpg")
imgChris = cv2.cvtColor(imgChris,cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("ImagesBasic/chris3.jpg")
imgTest = cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgChris)[0]
encodeChris = face_recognition.face_encodings(imgChris)[0]
cv2.rectangle(imgChris,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeChris],encodeTest)
faceDis = face_recognition.face_distance([encodeChris],encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results}{round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow("Chris Hermsworth",imgChris)
cv2.imshow("Chris Hermsworth Test",imgTest)
cv2.waitKey(0)