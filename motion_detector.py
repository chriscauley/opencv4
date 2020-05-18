import cv2
import numpy as np

FILE = "course_materials/Motion Detector/cars.mp4"
# FILE = 0

cap = cv2.VideoCapture(FILE)
ret, frame1 = cap.read()
frame0 = frame1
_frame2 = cap.read()

_gray = lambda frame: cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
_blur = lambda frame: cv2.GaussianBlur(_gray(frame), (21, 21), 0)

frame0_blur = frame1_blur = _blur(frame1)

while True:
    frame2 = frame1
    frame2_blur = frame1_blur
    frame1_blur = _blur(frame1)
    ret, frame1 = cap.read()

    if not ret:
        break

    diff = cv2.absdiff(frame1_blur,frame2_blur)

    thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)[1]
    final = cv2.dilate(thresh, None, iterations=2)
    masked = cv2.bitwise_and(frame1, frame1, mask=thresh)

    cv2.imshow("Frame", masked)
    #cv2.imshow("Frame", frame1)
    #cv2.imshow("Motion", diff)
    #cv2.imshow("Thresh", thresh)
    # cv2.imshow("Final", final)
    #cv2.imshow("Gray", _gray(frame1))
    #cv2.imshow("Blur", frame1_blur)

    key = cv2.waitKey(50)
    if key == ord('q'):
        break

cv2.destroyAllWindows()