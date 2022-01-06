import numpy as np
import mediapipe as mp
import cv2

############################ Func ############################


def GetPoint(landmarks):
    arr = np.zeros((21, 3))
    for ind, mark in enumerate(landmarks.landmark):
        arr[ind, 0] = mark.x
        arr[ind, 1] = mark.y
        arr[ind, 2] = mark.z

    return arr


def Knife(img, arr, l):
    if l != 0:
        h = 480
        w = 640
        s = np.array((w, h, 0))

        p1 = ((arr[5] + arr[9]) / 2)
        p2 = ((arr[9] + arr[13]) / 2)
        p3 = ((arr[13] + arr[17]) / 2)
        o = arr[0]

        v1 = p1 - o
        v2 = p2 - o
        v3 = p3 - o

        # 角度直一點 跟中間的平衡一下
        v1 = v1 * 2 + v2
        v3 = v3 * 2 + v2

        # 單位向量
        uv1 = v1 / (v1**2).sum()**0.5
        uv2 = v2 / (v2**2).sum()**0.5
        uv3 = v3 / (v3**2).sum()**0.5

        # 視覺比例
        r1 = (uv1[:2]**2).sum()**0.5
        r2 = (uv2[:2]**2).sum()**0.5
        r3 = (uv3[:2]**2).sum()**0.5

        # 螢幕大小要轉換像素了
        p1 = p1 * s
        p2 = p2 * s
        p3 = p3 * s

        color = (0, 0, 0)  # 刀的顏色
        # color = (20, 75, 125) # 香的顏色
        t = 2  # 爪粗細

        cv2.line(img, p1.astype(int)[:2], (p1 +
                 (uv1 * l * r1)).astype(int)[:2], color, t)
        cv2.line(img, p2.astype(int)[:2], (p2 +
                 (uv2 * l * r2)).astype(int)[:2], color, t)
        cv2.line(img, p3.astype(int)[:2], (p3 +
                 (uv3 * l * r3)).astype(int)[:2], color, t)

        # 上香版本
        # cv2.circle(img, (p1 + (uv1 * l * r1)).astype(int)[:2], 3, (0, 0, 225), -1)
        # cv2.circle(img, (p2 + (uv2 * l * r2)).astype(int)[:2], 3, (0, 0, 225), -1)
        # cv2.circle(img, (p3 + (uv3 * l * r3)).astype(int)[:2], 3, (0, 0, 225), -1)


############################ Main ############################

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(False, 2)


L = 180  # 爪長
t = 12  # 分幾禎伸縮

l = 0
add = 0
while True:
    ret, img = cap.read()
    img = img[:, ::-1].copy()  # 鏡像
    if ret:
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(imgRGB)

        if result.multi_hand_landmarks:
            l += add

            if l > L:
                l = L
            if l < 0:
                l = 0

            for handLms in result.multi_hand_landmarks:
                arr = GetPoint(handLms)
                Knife(img, arr, l)

        cv2.imshow('img', img)

    input_ = cv2.waitKey(5)

    if input_ == ord('q'):  # 關掉
        break

    elif input_ == ord('z'):  # 爪子縮放

        if add > 0:
            add = -(L / t)
        else:
            add = (L / t)

cap.release()
cv2.destroyAllWindows()
