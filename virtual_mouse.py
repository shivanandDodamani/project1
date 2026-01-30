import cv2
import numpy as np
import pyautogui
import math
import time
import mediapipe as mp

# Disable FailSafe to prevent crashes when mouse hits corners
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

# ================= Configuration =================
wCam, hCam = 640, 480       # Camera Resolution
frameR = 120                # Frame Reduction (margin) - Increased for faster cursor movement
smoothening = 3             # Smoothing Factor (Lower = faster response, less lag)
click_threshold = 28        # Distance threshold for click
# =================================================

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,              # 0 = Lite (Faster), 1 = Full
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,    # Lower confidence for faster detection
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Screen Size
wScr, hScr = pyautogui.size()

# Variables
pLocX, pLocY = 0, 0     # Previous Location
cLocX, cLocY = 0, 0     # Current Location
pTime = 0
left_down_th, left_up_th = 28, 38
right_down_th, right_up_th = 28, 38
hold_frames = 3
left_hold = 0
right_hold = 0
left_clicked = False
right_clicked = False

# Capture Device
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

print("Virtual Mouse Started.")
print("- Index Finger Up: Move Cursor")
print("- Index + Thumb Pinch: Left Click")
print("- Middle + Thumb Pinch: Right Click")
print("- Index + Middle Fingers Up: Scroll Mode")
print("Press 'q' to exit.")

try:
    while True:
        # 1. capture image
        success, img = cap.read()
        if not success:
            break

        # 2. Find Hand Landmarks
        img = cv2.flip(img, 1) # Mirror for natural interaction
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                lmList = []
                for id, lm in enumerate(hand_landmarks.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append([id, cx, cy])

                # Get coordinates of tips
                if len(lmList) != 0:
                    x1, y1 = lmList[8][1:]   # Index Finger Tip
                    x2, y2 = lmList[12][1:]  # Middle Finger Tip
                    x_thumb, y_thumb = lmList[4][1:] # Thumb Tip

                    # Check which fingers are up
                    fingers = []
                    # Thumb (simple check: if x of thumb is to the left of x of base(2) for right hand... 
                    # easier: check if tip is 'higher' than knuckle? No, thumb moves sideways.
                    # Simplified: just checking other 4 fingers for modes, thumb for clicks via distance.
                    
                    # Index Finger Up
                    fingers.append(1 if lmList[8][2] < lmList[6][2] else 0)
                    # Middle Finger Up
                    fingers.append(1 if lmList[12][2] < lmList[10][2] else 0)
                    
                    # --- Logic ---

                    # Mode 1: Moving (Only Index Finger Up)
                    if fingers[0] == 1 and fingers[1] == 0:
                        # Convert Coordinates
                        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
                        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))
                        
                        # Smoothen Values
                        cLocX = pLocX + (x3 - pLocX) / smoothening
                        cLocY = pLocY + (y3 - pLocY) / smoothening
                        
                        # Move Mouse
                        pyautogui.moveTo(cLocX, cLocY)
                        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                        pLocX, pLocY = cLocX, cLocY

                    # Mode 2: Clicking (Index and Thumb are close)
                    # Calculate distance between Index and Thumb
                    length_left = math.hypot(x1 - x_thumb, y1 - y_thumb)
                    if length_left < left_down_th:
                        left_hold += 1
                    elif length_left > left_up_th:
                        left_hold = 0
                        left_clicked = False
                    if left_hold >= hold_frames and not left_clicked:
                        cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                        pyautogui.click()
                        left_clicked = True

                    # Right Click (Middle and Thumb are close)
                    length_right = math.hypot(x2 - x_thumb, y2 - y_thumb)
                    if length_right < right_down_th:
                         right_hold += 1
                    elif length_right > right_up_th:
                         right_hold = 0
                         right_clicked = False
                    if right_hold >= hold_frames and not right_clicked:
                         cv2.circle(img, (x2, y2), 15, (0, 255, 0), cv2.FILLED)
                         pyautogui.rightClick()
                         right_clicked = True

                    # Mode 3: Scrolling (Index and Middle both UP)
                    if fingers[0] == 1 and fingers[1] == 1:
                        # Determine scroll based on distance between fingers? 
                        # No, usually based on movement. 
                        # Or position? Let's try simple:
                        # Map Y position of fingers to scroll. 
                        # But that's hard to control.
                        # Static gesture: If both up, we are in scroll mode.
                        # Let's verify distance between them to ensure it is deliberate.
                        dist_fingers = math.hypot(x1 - x2, y1 - y2)
                        if dist_fingers > 40: # If fingers are not pinched together
                            # Scroll sensitive to vertical movement? 
                            # Or just scroll if hands are in upper/lower part of box.
                            # Let's try: Scroll amount depends on fingertip Y position relative to center?
                            # Center of frame
                            # cy_center = hCam // 2
                            # if y1 < cy_center - 50:
                            #     pyautogui.scroll(100)
                            # elif y1 > cy_center + 50:
                            #     pyautogui.scroll(-100)
                            
                            # Better approach: track movement since last frame?
                            # Too jittery.
                            # Let's go with absolute position mapping for scroll speed.
                            
                            # Use simple linear mapping for now or a fixed scroll if fingers move.
                            # Let's stick to the request: "Scroll: Two-finger movement"
                            # I will implement: Scroll up if finger tips are HIGH, scroll down if LOW.
                            # Relative to the active frame.
                            
                            if y1 < hCam / 2 - 50:
                                pyautogui.scroll(20) # Scroll Up
                                cv2.putText(img, "Scroll Up", (20, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
                            elif y1 > hCam / 2 + 50:
                                pyautogui.scroll(-20) # Scroll Down
                                cv2.putText(img, "Scroll Down", (20, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
                            else:
                                pass # Neutral zone

        # Frame Rate
        cTime = time.time()
        fps = 1 / (cTime - pTime) if (cTime - pTime) > 0 else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

        cv2.imshow("Virtual Mouse", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
