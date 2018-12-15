import cv2
import numpy as np

def squarifinator(startX, startY, endX, endY):
    width = endX - startX
    height = endY - startY
    if width == height:
        return startX, startY, endX, endY
    elif height > width:
        tmp = int((height - width)/2)
        startX -= tmp
        endX += tmp
    else:
        tmp = int((width - height)/2)
        startY -= tmp
        endY += tmp
    width = endX - startX
    height = endY - startY
    if width > height:
        startY += 1
    elif width < height:
        endX += 1
    return (startX, startY, endX, endY)
        
        

class face_detector:
    def __init__(self, conf_limit=0.8):
        self.net = cv2.dnn.readNetFromCaffe("./model/deploy.prototxt", "./model/detect_model.caffemodel")
        self.conf_limit = conf_limit
    def extract_imgs(self, img):
        faces = []
        frame = img
        (h, w) = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)

        detections = self.net.forward()

        for i in range(0, detections.shape[2]):
            conf = detections[0, 0, i, 2]

            if conf < self.conf_limit:
                continue
            
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY, endX, endY) = squarifinator(startX, startY, endX, endY)

            face = frame[startY:endY, startX:endX]
            try:
                face = cv2.resize(face, (300, 300))
            except:
                continue
            faces.append(face)
        return faces