import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("VID19700113-203028.mp4")
#cap = cv2.VideoCapture("eye_recording.flv")
#eye_recording.flv

while True:
    ret, frame = cap.read()
    if ret is False:
        break

    roi = frame[200: 400, 100: 400]

    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)

    _, threshold = cv2.threshold(gray_roi, 75, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)

        #cv2.drawContours(roi, [cnt], -1, (0, 0, 255), 3)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
        break
    '''
    plt.subplot(1,3,1)
    plt.imshow(threshold)
    plt.subplot(1,3,2)
    plt.imshow(gray_roi)
    plt.subplot(1,3,3)
    plt.imshow(roi)
    plt.show()
    '''


    cv2.imshow("Threshold", threshold)
    cv2.imshow("gray roi", gray_roi)
    cv2.imshow("Roi", roi)
    key = cv2.waitKey(1000)
    if key == 27:
        break
cv2.destroyAllWindows()



