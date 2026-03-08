"""
crossy_road_gestures.py

Point with your index finger to move the character.
Open palm restarts the game (presses Enter).

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

DIR_THRESHOLD = 0.75
LR_THRESHOLD = 0.75

# normalize vector (and its directional components) to unit length, returns (0,0) if magnitude is too small
# this helps make direction detection more consistently regardless of hand distance from camera
# also allows us to define a single threshold for all directions based on cosine similarity
def unit(vx, vy):
    mag = (vx * vx + vy * vy) ** 0.5
    if mag < 1e-6:
        return 0.0, 0.0
    return vx / mag, vy / mag


def dist(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    return (dx * dx + dy * dy) ** 0.5


def index_extended(lm):
    # check if index finger is extended based on finger length
    tip = lm[8]
    base = lm[5]
    mid = lm[6]

    return dist(tip, base) > dist(mid, base) * 1.35


def open_palm(lm):
    # open palm = all four fingers extended
    index_ext = index_extended(lm)
    middle_ext = lm[12].y < lm[10].y
    ring_ext = lm[16].y < lm[14].y
    pinky_ext = lm[20].y < lm[18].y

    return index_ext and middle_ext and ring_ext and pinky_ext


def thumb_direction(lm):
    # use wrist -> thumb tip vector to determine direction (left or right)
    vx = lm[4].x - lm[0].x
    vy = lm[4].y - lm[0].y

    ux, uy = unit(vx, vy)

    scores = {
        "right": ux,
        "left": -ux,
        "down": uy,
        "up": -uy
    }

    direction = max(scores, key=scores.get)
    
    # If the dominant direction is horizontal, return it
    if direction in ("left", "right"):
        return direction, scores[direction]
        
    # If the dominant direction is vertical, ignore it for the thumb
    return "none", 0.0
    
def index_direction(lm):
    # use wrist -> index tip vector
    vx = lm[8].x - lm[5].x
    vy = lm[8].y - lm[5].y
    vz = lm[8].z - lm[5].z
    
    mag = (vx * vx + vy * vy + vz * vz) ** 0.5
    if mag < 1e-6:  
        return "none", 0.0
        
    ux = vx / mag
    uy = vy / mag  
    uz = vz / mag

    scores = {
        "up": -uz,
        "down": uy,
        "right": ux,
        "left": -ux
    }

    direction = max(scores, key=scores.get)
    
    # If the dominant direction is forward or backward, return it
    if direction in ("up", "down"):
        return direction, scores[direction]
        
    # If the dominant direction is horizontal, ignore it for the index finger
    return "none", 0.0
    
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
        pyautogui.click()

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

        # Used gemini here to modify the existing loop from the previous code to work with the new direction/gesture functions
        if result.hand_landmarks:

            lm = result.hand_landmarks[0]

            if open_palm(lm):

                status = "Detected: PALM"

                if controller.can_restart():
                    controller.restart()
                    status = controller.status

            else:
                # 1. Calculate directions for both fingers independently
                idx_dir, idx_score = index_direction(lm)
                thumb_dir, thumb_score = thumb_direction(lm)

                # 2. Prioritize Index finger (Forward / Backward)
                if idx_dir in ("up", "down") and idx_score >= DIR_THRESHOLD:
                    status = f"INDEX {idx_dir.upper()} ({idx_score:.2f})"
                    
                    if controller.can_move():
                        controller.move(idx_dir)
                        status = controller.status

                # 3. If the index finger isn't triggering a move, check the Thumb (Left / Right)
                elif thumb_dir in ("left", "right") and thumb_score >= LR_THRESHOLD:
                    status = f"THUMB {thumb_dir.upper()} ({thumb_score:.2f})"
                    
                    if controller.can_move():
                        controller.move(thumb_dir)
                        status = controller.status

                else:
                    status = "Hand seen (Waiting for gesture)"

        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

        cv2.putText(frame,
                    "Point to move | Open palm = restart | ESC to quit",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200), 2)

        cv2.imshow("Crossy Road Gestures (ESC to quit)", frame)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()