"""
crossy_road_gestures.py

Custom Crossy Road gesture controls:
- Thumb out = move left
- Pointer finger only = move up
- Pointer + middle finger = move down
- Pinky only = move right
- Open palm restarts the game (presses Enter)

Requires:
hand_landmarker.task
"""

import time
import webbrowser

import cv2
import mediapipe as mp
import pyautogui

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode


GAME_URL = "https://crossyroadgame.io/"
AUTO_OPEN_BROWSER = True
HAND_MODEL = "hand_landmarker.task"

FRAME_W = 640
FRAME_H = 480

MOVE_COOLDOWN = 0.18
RESTART_COOLDOWN = 0.70


def finger_extended(lm, tip_idx, pip_idx):
    # checks if a finger is extended by comparing tip and middle joint
    return lm[tip_idx].y < lm[pip_idx].y


def open_palm(lm):
    # open palm = all four fingers extended
    index_ext = finger_extended(lm, 8, 6)
    middle_ext = finger_extended(lm, 12, 10)
    ring_ext = finger_extended(lm, 16, 14)
    pinky_ext = finger_extended(lm, 20, 18)

    return index_ext and middle_ext and ring_ext and pinky_ext


def thumb_out(lm):
    # thumb out = thumb extended sideways while other fingers stay folded
    thumb_tip = lm[4]
    thumb_ip = lm[3]
    index_base = lm[5]

    index_folded = lm[8].y > lm[6].y
    middle_folded = lm[12].y > lm[10].y
    ring_folded = lm[16].y > lm[14].y
    pinky_folded = lm[20].y > lm[18].y

    thumb_side = abs(thumb_tip.x - index_base.x) > 0.08
    thumb_extended = abs(thumb_tip.x - thumb_ip.x) > 0.03

    return thumb_side and thumb_extended and index_folded and middle_folded and ring_folded and pinky_folded


def pointer_only(lm):
    # only index finger extended
    index_ext = finger_extended(lm, 8, 6)
    middle_ext = finger_extended(lm, 12, 10)
    ring_ext = finger_extended(lm, 16, 14)
    pinky_ext = finger_extended(lm, 20, 18)

    return index_ext and not middle_ext and not ring_ext and not pinky_ext


def pointer_middle(lm):
    # index and middle finger extended together
    index_ext = finger_extended(lm, 8, 6)
    middle_ext = finger_extended(lm, 12, 10)
    ring_ext = finger_extended(lm, 16, 14)
    pinky_ext = finger_extended(lm, 20, 18)

    return index_ext and middle_ext and not ring_ext and not pinky_ext


def pinky_only(lm):
    # only pinky extended
    index_ext = finger_extended(lm, 8, 6)
    middle_ext = finger_extended(lm, 12, 10)
    ring_ext = finger_extended(lm, 16, 14)
    pinky_ext = finger_extended(lm, 20, 18)

    return pinky_ext and not index_ext and not middle_ext and not ring_ext


class Controller:

    def __init__(self):
        self.last_move = 0
        self.last_restart = 0
        self.status = "Starting..."

    def can_move(self):
        return time.time() - self.last_move > MOVE_COOLDOWN

    def can_restart(self):
        return time.time() - self.last_restart > RESTART_COOLDOWN

    def move(self, key):
        pyautogui.press(key)
        self.last_move = time.time()
        self.status = f"MOVE -> {key.upper()}"

    def restart(self):
        pyautogui.press("enter")
        self.last_restart = time.time()
        self.status = "PALM -> ENTER (restart)"


def main():

    if AUTO_OPEN_BROWSER:
        webbrowser.open(GAME_URL)
        time.sleep(2)
        pyautogui.click()  # focuses the browser/game window

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=RunningMode.VIDEO,
        num_hands=1
    )

    hand_landmarker = HandLandmarker.create_from_options(options)

    controller = Controller()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    while True:

        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp = int(time.time() * 1000)
        result = hand_landmarker.detect_for_video(mp_image, timestamp)

        status = "No hand"

        if result.hand_landmarks:

            lm = result.hand_landmarks[0]

            if open_palm(lm):
                status = "Detected: PALM"

                if controller.can_restart():
                    controller.restart()
                    status = controller.status

            elif thumb_out(lm):
                status = "Detected: THUMB -> LEFT"

                if controller.can_move():
                    controller.move("left")
                    status = controller.status

            elif pointer_middle(lm):
                status = "Detected: POINTER + MIDDLE -> DOWN"

                if controller.can_move():
                    controller.move("down")
                    status = controller.status

            elif pointer_only(lm):
                status = "Detected: POINTER -> UP"

                if controller.can_move():
                    controller.move("up")
                    status = controller.status

            elif pinky_only(lm):
                status = "Detected: PINKY -> RIGHT"

                if controller.can_move():
                    controller.move("right")
                    status = controller.status

            else:
                status = "Hand seen"

        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

        cv2.putText(frame,
                    "Thumb=Left | Pointer=Up | Pointer+Middle=Down | Pinky=Right | Palm=Restart",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (200, 200, 200), 2)

        cv2.imshow("Crossy Road Gestures (ESC to quit)", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()