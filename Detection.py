import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import math
from pynput.keyboard import Key, Controller

DEBUG = False 

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.2, min_tracking_confidence=0.2)


cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
keyboard = Controller()



def detectPose(frame, pose):
    output_frame = frame.copy()
    imageRGB = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
    result = pose.process(imageRGB)

    height, width, depth = frame.shape
    landmarks = []

    if result.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=output_frame, landmark_list=result.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
        for landmark in result.pose_landmarks.landmark:
            landmarks.append(
                (int(landmark.x * width), int(landmark.y * height), int(landmark.z * depth)))
    return output_frame, landmarks

# Define Pose Classification Function


def classifyPose(frame, pose):
    label = '[]'

    def calculateAngle(l1, l2, l3):
        x1, y1, z1 = l1
        x2, y2, z2 = l2
        x3, y3, z3 = l3
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360
        return angle

    def isAngleBetween(arg, n1, n2):
        if n1 > n2:
            if arg >= n1:
                return True
            else:
                return arg <= n2
        return n1 <= arg <= n2

    try:
        left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
        right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
        left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
        right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])

        if DEBUG:
            label = 'LElbow: ' + str(round(left_elbow_angle)) + \
                '/ RElbow: ' + str(round(right_elbow_angle)) + \
                '/ LShoulder: ' + str(round(left_shoulder_angle)) + \
                '/ RShoulder: ' + str(round(right_shoulder_angle))
            print(label)

        # The Mathematical Definition of a Dab
        if isAngleBetween(left_elbow_angle, 160, 230):
            if isAngleBetween(right_elbow_angle, 280, 50):
                if isAngleBetween(left_shoulder_angle, 50, 120) and isAngleBetween(right_shoulder_angle, 230, 300):
                    label = 'abe jump kar!!'
                    keyboard.press(Key.space)
                    keyboard.release(Key.space)
        if isAngleBetween(right_elbow_angle, 160, 230):
            if isAngleBetween(left_elbow_angle, 280, 50):
                if isAngleBetween(right_shoulder_angle, 230, 300) and isAngleBetween(left_shoulder_angle, 50, 120):
                    label = 'koi kam ka nhi hai tu!!'
                    keyboard.press(Key.space)
                    keyboard.release(Key.space)
    except IndexError:
        pass
    return label


# Get and Show Webcam Frame
while cam.isOpened():
    res, frame = cam.read()
    if not res:
        print('Error.', res)
        break

    output_frame, landmarks = detectPose(frame, pose)

    label = classifyPose(output_frame, pose)
    font = cv2.FONT_HERSHEY_PLAIN
    scale = 3
    thinkness = 5
    label_size = cv2.getTextSize(label, font, scale, thinkness)[0]
    label_pos = (int((output_frame.shape[1] - label_size[0]) / 2), 80)

    cv2.putText(output_frame, label, org=label_pos, lineType=cv2.LINE_AA,
                fontFace=font, fontScale=scale, color=[0, 0, 0], thickness=thinkness)
    cv2.putText(output_frame, label, org=label_pos, lineType=cv2.LINE_AA,
                fontFace=font, fontScale=scale, color=[255, 255, 255], thickness=thinkness - 3)
    cv2.imshow('Pose Detection', output_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
