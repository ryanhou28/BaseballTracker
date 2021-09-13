import cv2
import numpy as np
import imutils

#Create Video Capture

#   Video:
cap = cv2.VideoCapture("Videos/Video3.mov")
cnt = 0
#   Webcam:
#cap = cv2.VideoCapture(0)

if (cap.isOpened() == False):
    print("Error opening video!")

ret, firstFrame = cap.read()


#Main video loop
while (cap.isOpened()):

    #Capture frame
    ret, frame = cap.read()

    if ret == True:
        #If a return value is captured

        #Crop:
        roi = frame[:, :] #No crop for now

        #Cropping Center of image
        thresh = 600
        end = roi.shape[1] - thresh
        roi = roi[:, thresh:end]

        cv2.imshow("Frame", roi)

        #Press Q to exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

        cv2.imwrite("Frames/" + str(cnt) + '.png', roi)
        cnt += 1

    else:
        break


cv2.destroyAllWindows()
cap.release()
