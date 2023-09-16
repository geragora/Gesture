import autopy
import cv2
import mediapipe
import numpy as np
import time
import HandTrackingModule as htm

plocX, plocY = 0, 0
clocX, clocY =0, 0

cTime = 0
pTime = 0
wCam, hCam = 640, 480
frameR = 100 #frame rediction
smoothening = 5
cap = cv2.VideoCapture(1)
cap.set(3,wCam)
cap.set(4,hCam)

detector = htm.handDetector(maxHands= 1)
wScr, hScr = autopy.screen.size()
while True:

    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw= False)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1,y1,x2,y2)
        # print(lmList[4])
        fingers = detector.fingersUp(lmList)
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)#приподнять вверх чтобы не вылеитала
        if fingers[1] ==1 and fingers[2] == 0:


            x3 = np.interp(x1, (frameR,wCam-frameR), (0,wScr))
            y3 = np.interp(y1, (frameR,hCam-frameR), (0,hScr))


            clocX = plocX +(x3 - plocX) / smoothening
            clocY = plocY +(y3 - plocY) / smoothening

            autopy.mouse.move(wScr-clocX, clocY)
            cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED)
            plocx, plocY = clocX, clocY
        if fingers[1] ==1 and fingers[2] == 1:
            length,img, info = detector.findDistance(8,12, img)
           # print(length) говорит нормализацию и тут можнго провести
            if length<40 :
                cv2.circle(img, (info[4],info[5]), 15, (0,255, 0), cv2.FILLED)
                autopy.mouse.click()
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20,50), cv2.FONT_HERSHEY_PLAIN, 3,(255,0,0),3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)