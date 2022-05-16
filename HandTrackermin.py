import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

# creating own model
mpHand = mp.solutions.hands
hands = mpHand.Hands()
mpDraw = mp.solutions.drawing_utils

#FPS
#previoustime
pTime = 0
#currenttime
cTime = 0

while True:
    success, img = cap.read()
    #importing RGB image

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #objects with image results
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            #landmark information to get coordinates of x,y and z we will use x and y for landmark

            for id, lm in enumerate(handlms.landmark):
                print(id, lm)
            mpDraw.draw_landmarks(img, handlms, mpHand.HAND_CONNECTIONS)

    cTime = time.time()
    fps = int(1/(cTime-pTime))
    pTime = cTime

    cv2.putText(img, str(fps), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 255), 3)

    cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Image', img)
    cv2.waitKey(1)








#
# import cv2
# import numpy as np
# video = cv2.VideoCapture(0)
# if video.isOpened():
#     for i in range(int(1e12)):
#         check, frame = video.read()
#
#         if check:
#             cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)
#             cv2.imshow('frame', frame)
#             cv2.waitKey(1)
#             cv2.destroyAllWindows()
#             #cv2.imshow("image", frame)
#             #cv2.imwrite(f'{i}.png', np.array(frame))
#         else:
#             print('Frame not available')
#             print(video.isOpened())
# else:
#     print("video not opened")