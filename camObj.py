# ----------------------------------------------------------------------------
# camInfo.py
# Author: Bhargav Bhalodi
# Description: Camera Object class to retrieve video information from a camera
# device.
# ----------------------------------------------------------------------------

# Camera Object class
# Creates camera objects for retrieving video information/data from a camera
# device.
import numpy as np
import cv2

class CamObj:

    def __init__(self):
        # variable 1 - holder for camera object
        # variable 2 - holder for video data
        pass

    def set_cam(self, newCam):
        # populates the camera object variable with a camera device
        # using OpenCV
        pass

    def initialize_video_data(self, camIndex=0, show=False, needRet=False,
                              filename=''):
        # initializes video capture from camera at index = camIndex.
        # displays the capture if show is set to True.
        # press 'q' to quit the display.
        # if video data needs to be returned, set needRet to be True and
        # filename should be provided in the form of fname.avi
        beginCap = cv2.VideoCapture(camIndex)
        if needRet:
            vidWriter = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
            outputDestination = cv2.VideoWriter(filename, vidWriter,
                                                60.0, (640, 480))
        while True:
            check, frames = beginCap.read()
            if show:
                cv2.imshow('frame', frames)
            if needRet:
                outputDestination.write(frames)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        beginCap.release()
        if needRet:
            outputDestination.release()
        cv2.destroyAllWindows()


myCam = CamObj()
myCam.initialize_video_data(0, show=True, needRet=True, filename='trial.avi')
