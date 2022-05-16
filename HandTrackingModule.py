import cv2
import mediapipe as mp
import time


class HandDetector:

    def __init__(self, mode=False, maxhands=2, modcomp=1, detectconf=0.5, trackconf=0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.modcomp = modcomp
        self.detectconf = detectconf
        self.trackconf = trackconf

        # creating own model
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxhands, self.modcomp, self.detectconf, self.trackconf)
        self.mpDraw = mp.solutions.drawing_utils

    # function to get for one particular hand
    def findHands(self, img, draw=True):
        # importing RGB image
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # objects with image results
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handlms in self.results.multi_hand_landmarks:
                # landmark information to get coordinates of x,y and z we will use x and y for landmark
                if draw:
                    self.mpDraw.draw_landmarks(img, handlms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []

        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)
                lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        # if id == 2:
        #     cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FILLED)
        return lmList


def main():
    #FPS
    #previoustime
    pTime = 0
    #currenttime
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = (1 / (cTime - pTime))
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (255, 0, 255), 3)
        cv2.namedWindow('Image', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
