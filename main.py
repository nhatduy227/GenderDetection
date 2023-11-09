from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import cvlib as cv

# load model
model = load_model("gender_detection.model")
classes = ["man", "woman"]

webcam = cv2.VideoCapture(0)
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.moveWindow("Image", 0, 0)
cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def setup():
    malePath = os.path.normpath(os.path.expanduser("~/Desktop/GenderDetection/male"))
    # malePath = os.path.normpath(
    #     os.path.expanduser("~/OneDrive/Desktop/GenderDetection/male")
    # )
    maleVideo = "/" + os.listdir(malePath)[0]
    male = cv2.VideoCapture(malePath + maleVideo)

    femalePath = os.path.normpath(
        os.path.expanduser("~/Desktop/GenderDetection/female")
    )
    # femalePath = os.path.normpath(
    #     os.path.expanduser("~/OneDrive/Desktop/GenderDetection/female")
    # )
    femaleVideo = "/" + os.listdir(femalePath)[0]
    female = cv2.VideoCapture(femalePath + femaleVideo)

    defaultPath = os.path.normpath(
        os.path.expanduser("~/Desktop/GenderDetection/default")
    )
    # defaultPath = os.path.normpath(
    #     os.path.expanduser("~/OneDrive/Desktop/GenderDetection/default")
    # )
    defaultVideo = "/" + os.listdir(defaultPath)[0]
    default = cv2.VideoCapture(defaultPath + defaultVideo)
    return male, female, default


def loopVideo(status, cap, frame, gender):
    if status:
        if gender == "male" or gender == "woman":
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            cv2.imshow("Image", frame)
        else:
            cv2.imshow("Image", frame)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


# OBSOLETE CODE TO BE REMOVE SOON


def detectionCamOldVersion():
    try:
        padding = 20
        gender = ""
        face, confidence = cv.detect_face(frame)
        for idx, f in enumerate(face):
            (startX, startY) = max(0, f[0] - padding), max(0, f[1] - padding)
            (endX, endY) = min(frame.shape[1] - 1, f[2] + padding), min(
                frame.shape[0] - 1, f[3] + padding
            )
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            face_crop = np.copy(frame[startY:endY, startX:endX])

            (label, confidence) = cv.detect_gender(face_crop)
            idx = np.argmax(confidence)
            label = label[idx]
            gender = label

            label = "{}: {:.2f}%".format(label, confidence[idx] * 100)
            Y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.putText(
                frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        return gender
    except:
        print("Wrong image input")


def detectionCam(frame):
    try:
        gender = ""
        # apply face detection
        face, confidence = cv.detect_face(frame)

        # loop through detected faces
        for idx, f in enumerate(face):
            # get corner points of face rectangle
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            # draw rectangle over face
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

            # crop the detected face region
            face_crop = np.copy(frame[startY:endY, startX:endX])

            if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                continue

            # preprocessing for gender detection model
            face_crop = cv2.resize(face_crop, (96, 96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            # apply gender detection on face
            conf = model.predict(face_crop)[
                0
            ]  # model.predict return a 2D matrix, ex: [[9.9993384e-01 7.4850512e-05]]

            # get label with max accuracy
            idx = np.argmax(conf)
            label = classes[idx]
            gender = label

            label = "{}: {:.2f}%".format(label, conf[idx] * 100)

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            # write label and confidence above face rectangle
            cv2.putText(
                frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            return gender
    except:
        print("Wrong image input")


def mainProcess():
    male, female, default = setup()
    while webcam.isOpened():
        status, frame = webcam.read()
        maleStatus, maleFrame = male.read()
        femaleStatus, femaleFrame = female.read()
        defaultStatus, defaultFrame = default.read()
        gender = detectionCam(frame)

        # if status:
        #     cv2.imshow("Real-time gender detection", frame)

        if gender == "man":
            loopVideo(maleStatus, male, maleFrame, gender)
        elif gender == "woman":
            loopVideo(femaleStatus, female, femaleFrame, gender)
        else:
            loopVideo(defaultStatus, default, defaultFrame, gender)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    webcam.release()
    cv2.destroyAllWindows()


mainProcess()
