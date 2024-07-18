import cv2
from cvzone.HandTrackingModule import HandDetector

video_capture = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.5, maxHands=2)
while True:
    ret, frame = video_capture.read()

    hand, img = detector.findHands(frame)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
