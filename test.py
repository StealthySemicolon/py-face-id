import cv2
import numpy as np
from tqdm import trange
from detect_faces import face_detector

cap = cv2.VideoCapture(0)
detector = face_detector(0.2)

ret, img = cap.read()

for i in trange(10000):
    ret, img = cap.read()
    faces = detector.extract_imgs(img)
    try:
        cv2.imshow("img", faces[0])
    except:
        pass
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break