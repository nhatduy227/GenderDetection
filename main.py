import cvlib as cv
import cv2
import numpy as np
import os

webcam = cv2.VideoCapture(0)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.moveWindow("Image", 0,0)
cv2.setWindowProperty ("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def setup():
    malePath = os.path.normpath(os.path.expanduser("~/Desktop/GenderDetection/male"))
    maleVideo = "/" + os.listdir(malePath)[0]
    male = cv2.VideoCapture(malePath + maleVideo)

    femalePath = os.path.normpath(os.path.expanduser("~/Desktop/GenderDetection/female"))
    femaleVideo = "/" + os.listdir(femalePath)[0]
    female = cv2.VideoCapture(femalePath + femaleVideo)

    defaultPath = os.path.normpath(os.path.expanduser("~/Desktop/GenderDetection/default"))
    defaultVideo = "/" + os.listdir(defaultPath)[0]
    default = cv2.VideoCapture(defaultPath + defaultVideo)
    return male, female, default
    
def loopVideo(status, cap, frame):
    if status:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  
        cv2.imshow("Image", frame)
    else:
       cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

def detectionCam(): 
    padding = 20
    gender = ""
    face, confidence = cv.detect_face(frame)
    for idx, f in enumerate(face):        
        (startX,startY) = max(0, f[0]-padding), max(0, f[1]-padding)
        (endX,endY) = min(frame.shape[1]-1, f[2]+padding), min(frame.shape[0]-1, f[3]+padding)
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
        face_crop = np.copy(frame[startY:endY, startX:endX]) 
        (label, confidence) = cv.detect_gender(face_crop)
        idx = np.argmax(confidence)
        label = label[idx]
        gender = label
        # label = "{}: {:.2f}%".format(label, confidence[idx] * 100)
        # Y = startY - 10 if startY - 10 > 10 else startY + 10
        # cv2.putText(frame, label, (startX,Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
        #             (0,255,0), 2)
    return gender

male, female, default = setup()
while webcam.isOpened():
    status, frame = webcam.read()
    maleStatus, maleFrame = male.read()
    femaleStatus, femaleFrame = female.read()
    defaultStatus, defaultFrame = default.read()
    gender = detectionCam()

    # if status: 
    #     cv2.imshow("Real-time gender detection", frame)

    if gender == "male":
        loopVideo(maleStatus, male, maleFrame)
    elif gender == "female":
        loopVideo(femaleStatus, female, femaleFrame)
    else:
        loopVideo(defaultStatus, default, defaultFrame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()