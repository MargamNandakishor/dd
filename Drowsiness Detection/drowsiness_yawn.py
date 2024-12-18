from scipy.spatial import distance
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os


def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status:
        s = f'espeak "{msg}"'
        os.system(s)

    if alarm_status2:
        saying = True
        os.system(f'espeak "{msg}"')
        saying = False


def eye_aspect_ratio(eye):
    # Compute the Eye Aspect Ratio (EAR)
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def final_ear(shape):
    # Extract the left and right eye landmarks and compute the EAR
    leftEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]:
                    face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1]]
    rightEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]:
                     face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return ear, leftEye, rightEye


def lip_distance(shape):
    # Compute the distance between the lips
    top_lip = np.concatenate((shape[50:53], shape[61:64]))
    low_lip = np.concatenate((shape[56:59], shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    return abs(top_mean[1] - low_mean[1])


# Command line argument parsing
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="index of webcam on system")
args = vars(ap.parse_args())

# Constants
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

# Load face detector and shape predictor from dlib
print("-> Loading the predictor and detector...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Start video stream
print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)

while True:
    frame = vs.read()

    if frame is None:
        print("Failed to capture image from video stream")
        break

    # Resize frame and convert to grayscale
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))

        # Get facial landmarks
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Get eye aspect ratio and lip distance
        ear, leftEye, rightEye = final_ear(shape)
        distance = lip_distance(shape)

        # Draw the contours of the eyes and lips
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        # Drowsiness detection logic based on EAR
        if ear < EYE_AR_THRESH:
            COUNTER += 1

            if COUNTER >= EYE_AR_CONSEC_FRAMES and not alarm_status:
                alarm_status = True
                t = Thread(target=alarm, args=('Wake up sir!',))
                t.daemon = True
                t.start()

            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            COUNTER = 0
            alarm_status = False

        # Yawn detection logic based on lip distance
        if distance > YAWN_THRESH:
            if not alarm_status2 and not saying:
                alarm_status2 = True
                t = Thread(target=alarm, args=('Take some fresh air, sir!',))
                t.daemon = True
                t.start()

            cv2.putText(frame, "Yawn Alert!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            alarm_status2 = False

        # Display EAR and Yawn distance values
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"YAWN: {distance:.2f}", (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame with annotations
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
