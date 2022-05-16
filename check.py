import cv2
while True:
    cap = cv2.VideoCapture(0)
    success, img = cap.read()
    #cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Image', img)
    cv2.waitKey(1)