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

    def setCam(self, newCam):
        # populates the camera object variable with a camera device
        # using OpenCV
        pass

    def detectFace(self, frame, face_crds):
        # draws rectangles around faces at coordinates provided by face_crds
        #  in the frame given as frame.
        modified_frame = frame
        for (a,b,c,d) in face_crds:
            modified_frame = cv2.rectangle(frame, (a,b), (a+c,b+d), (0,0,0), 5)

        return modified_frame

    def initializeVideoData(self, cam_index=0, show=False, need_ret=False,
                              filename=''):
        # initializes video capture from camera at index = camIndex.
        # displays the capture if show is set to True.
        # press 'q' to quit the display.
        # if video data needs to be returned, set needRet to be True and
        # filename should be provided in the form of fname.avi
        begin_cap = cv2.VideoCapture(cam_index)
        face_detector = cv2.CascadeClassifier(
            'C:\Python3\Lib\site-packages\cv2\data'
            '\haarcascade_frontalface_default.xml')
        if need_ret:
            vid_writer = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
            output_destination = cv2.VideoWriter(filename, vid_writer,
                                                60.0, (640, 480))
        while True:
            check, frames = begin_cap.read()
            grayscale_frame = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
            face_coords = np.uint8(face_detector.detectMultiScale(
                frames, 1.1, 3))
            detected_frame = self.detectFace(frames, face_coords)
            if show:
                cv2.imshow('frame', detected_frame)
                #cv2.imshow('grayscale_frame', grayscale_frame)
            if need_ret:
                output_destination.write(detected_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        begin_cap.release()
        if need_ret:
            output_destination.release()
        cv2.destroyAllWindows()


myCam = CamObj()
myCam.initializeVideoData(0, show=True, need_ret=True, filename='trial.avi')
