# Demo: use OpenCV Haar cascade. Replace with MTCNN/Dlib later.
import cv2

def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade)
    faces = detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

def blur_regions(img, boxes):
    out = img.copy()
    for (x, y, w, h) in boxes:
        roi = out[y:y+h, x:x+w]
        if roi.size == 0: continue
        roi = cv2.GaussianBlur(roi, (51,51), 0)
        out[y:y+h, x:x+w] = roi
    return out
