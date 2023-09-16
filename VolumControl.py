import cv2
from cvzone.HandTrackingModule import HandDetector
import math, time, numpy as np
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
def findDistance(p1, p2, img=None):
    x1, y1 = p1
    x2, y2 = p2
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    length = math.hypot(x2 - x1, y2 - y1)
    info = (x1, y1, x2, y2, cx, cy)
    if img is not None:
        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        return length, info, img
    else:
        return length, info

cap = cv2.VideoCapture(1)
detector = HandDetector(detectionCon = 0.8, maxHands=2)
pTime = 0
length = 0
cTime = 0
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
while True:
    success, img = cap.read()
    hands, img  = detector.findHands(img)
    if len(hands) == 2:
            hand1 = hands[0]
            lmList1 = hand1["lmList"]#21 координата сустава
            bbox1 = hand1["bbox"]#координаты и ширина с высотой ящика с рукой
            centerPoint1 = hand1["center"]#координаты центра руки
            handType1 = hand1["type"]# левая или правая
            fingers1 = detector.fingersUp(hand1)
            # length, info,img = findDistance(lmList1[8][0:2],lmList1[12][0:2], img)
            # print(lmList1[8])
            hand2 = hands[1]
            lmList2 = hand2["lmList"]  # 21 координата сустава
            bbox2= hand2["bbox"]  # координаты и ширина с высотой ящика с рукой
            centerPoint2 = hand2["center"]  # координаты центра руки
            handType2 = hand2["type"]  # левая или правая
            fingers2 = detector.fingersUp(hand2)
            # print(fingers1,fingers2)
            length, info,img = findDistance(lmList1[8][0:2],lmList2[8][0:2], img)

            vol = np.interp(length, [30,250], [minVol,maxVol])
            volume.SetMasterVolumeLevel(vol, None)




    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
    cv2.putText(img, str(int(length)), (400, 40), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)