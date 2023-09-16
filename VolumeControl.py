import cv2
import time
import numpy as np
import HandTrackingModule as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

cap = cv2.VideoCapture(1)
pTime = 0
volBar = 0
vol =0
volPer =0
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]


detector =  htm .handDetector(detectionCon=0.7)
while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)

    if len (lmList)!= 0:
        x1,y1 = lmList[4][1],lmList[4][2]
        x2,y2 = lmList[8][1],lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2
        cv2.circle(img, (x1,y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1,y1), (x2,y2),(255,0,255),3)

        lenght = math.hypot(x2-x1,y2-y1)

        vol = np.interp(lenght, [30,250], [minVol,maxVol])
        volBar = np.interp(lenght, [30,250], [400,150])
        volPer = np.interp(lenght, [30,250], [0,100])

        volume.SetMasterVolumeLevel(vol, None)
        cv2.rectangle(img, (50,150), (85,400), (255,0,0), 3)
        cv2.rectangle(img, (50,int(volBar)), (85,400), (255,0,0), cv2.FILLED)


    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)
    cv2.putText(img, str(int(volPer)), (40, 450), cv2.FONT_HERSHEY_PLAIN, 3,
                (0, 250, 0), 3)
    cv2.imshow('Img', img)
    cv2.waitKey(1)

