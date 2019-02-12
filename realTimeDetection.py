# -----------------------------------------------------------------------------

# Help was taken from Adrian Rosebrock's codes for the following posts:

# https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep
# -learning-and-opencv/
# https://www.pyimagesearch.com/2017/09/18/real-time-object-detection-with
# -deep-learning-and-opencv/

# -----------------------------------------------------------------------------

# Help for the models was taken from the following sources:

# MobileNet SSD pre-trained model: Caffe version of the model by Howard et
# al at https://github.com/Zehaos/MobileNet.

# Bvlc Googlenet pre-trained model: Model at
# https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet by
# Christian Szegedy and Guadarrama. The model is a replication of the the
# model described in the GoogleNet paper found at
# https://arxiv.org/abs/1409.4842 by Szegedy et al.

# -----------------------------------------------------------------------------

# Usage directions:
# 1. Run the file.
# 2. Choose desired model when asked in the console. Currently supported
# model is only the default model. Working on adding new models.
# 3. Press 'q' to quit the video stream that pops up.

# import necessary modules
import cv2
import numpy as np
import imutils
import imutils.video
import time


# main runtime function
def main():
    # creates an instance of a real time detector object
    detector = RealTimeDetector()

    # prints information regarding the model currently in use
    display_model_info(detector)

    # options to change the detector model
    change_model = str(input("Would you like to switch to a different model? "
                             "(Y/N): "))
    if change_model == "Y":
        config_filename = str(input("Please input the configuration "
                                      "filename: "))
        model_filename = str(input("Please input the model filename: "))
        detector.set_detector_model(config_filename, model_filename)
        display_model_info(detector)

    # starts the video stream
    print("===> Starting video stream ...")
    stream = imutils.video.VideoStream(src=0).start()
    time.sleep(2.0)
    colors = np.random.uniform(0, 255, size=(len(detector.class_list),
                                             3))

    while True:
        # detects objects in the video stream
        current_frame = stream.read()
        # current_frame = imutils.resize(current_frame, width=400)
        (height, width) = current_frame.shape[:2]
        frame_detections = detector.get_detections(current_frame)
        for det_num in np.arange(0, frame_detections.shape[2]):
            class_index = int(frame_detections[0,0,det_num,1])
            bbox_coords = frame_detections[0,0,det_num,3:7]*np.array([width,
                                                                     height,
                                                                     width,
                                                                     height])
            x1, y1, x2, y2 = bbox_coords.astype("int")

            detection_label = "{}: {:.2f}%".format(detector.class_list[
                                                   class_index],
                                     frame_detections[0,0,det_num,2] * 100)
            cv2.rectangle(current_frame, (x1, y1), (x2, y2),
                          colors[class_index], 2)
            y = y1 - 15 if y1 - 15 > 15 else y1 + 15
            cv2.putText(current_frame, detection_label, (x1, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_index], 2)

        # displays the frame with detected objects if any
        cv2.imshow("Real Time Object Detection", current_frame)

        # quits the program if 'q' is pressed
        quit_key = cv2.waitKey(1) & 0xFF
        if quit_key == ord("q"):
            break


def display_model_info(detector_object):
    print("===> Displaying current model information:")
    print("         Model configuration filename: ",
          detector_object.detector_config_file)
    print("         Model filename: ",
          detector_object.detector_model_file)
    print("===> Detection possibilities for chosen model:")
    for i in detector_object.class_list:
        print(i)


# Real time object detector class definition
# Initializes to bvlc_googlenet object detection model
class RealTimeDetector:
    # initializes variables for real time object detector
    # defaults to bvlc_googlenet model
    def __init__(self):
        self.detector_config_file = "MobileNetSSD_deploy.prototxt.txt"
        self.detector_model_file = "MobileNetSSD_deploy.caffemodel"
        self.detector = cv2.dnn.readNetFromCaffe(
            self.detector_config_file, self.detector_model_file)
        self.class_list = self.update_class_list(self.detector_model_file)

    # changes detector model based on given model information
    def set_detector_model(self, config_filename, model_filename):
        self.detector_config_file = config_filename
        self.detector_model_file = model_filename

        # Darknet model setup
        if config_filename[-3:] == "cfg":
            self.detector = cv2.dnn.readNetFromDarknet(config_filename,
                                                       model_filename)
        # Caffe model setup
        else:
            self.detector = cv2.dnn.readNetFromCaffe(config_filename,
                                                 model_filename)
        self.class_list = self.update_class_list(model_filename)

    # updates the class list based on what dataset the model was trained on
    def update_class_list(self, model_filename):
        if model_filename == "MobileNetSSD_deploy.caffemodel":
            return ["background", "aeroplane", "bicycle", "bird", "boat",
                    "bottle", "bus", "car", "cat", "chair", "cow",
                    "diningtable", "dog", "horse", "motorbike", "person",
                    "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
        elif model_filename == "bvlc_googlenet.caffemodel":
            googlenet_classes_filename = "synset_words.txt"
            row_split = open(googlenet_classes_filename).read().strip().split(
                "\n")
            return [row[row.find(" ") + 1:].split(",")[0] for row in row_split]
        elif model_filename == "yolov3.weights":
            yolov3_classes_filename = "coco.names"
            return open(yolov3_classes_filename).read().strip().split("\n")

    # returns blob from image with parameters depending on the model in use
    def get_blob(self, video_frame):
        model_filename = self.detector_model_file
        video_frame = imutils.resize(video_frame, width=400)
        if model_filename == "bvlc_googlenet.caffemodel":

            frame_blob = cv2.dnn.blobFromImage(video_frame, 1, (224, 224),
                                               (104, 117, 123))
        if model_filename == "MobileNetSSD_deploy.caffemodel":
            frame_blob = cv2.dnn.blobFromImage(cv2.resize(video_frame, (300,
                                                                        300)),
                                               0.007843, (300, 300), 127.5)
        if model_filename == "yolov3.weights":
            frame_blob = cv2.dnn.blobFromImage(video_frame, 1 / 255.0,
                                               (416, 416), swapRB=True,
                                               crop=False)

        return frame_blob

    # returns detection data for a particular video frame based on the
    # current model in use
    def get_detections(self, video_frame):
        frame_blob = self.get_blob(video_frame)
        self.detector.setInput(frame_blob)
        det = self.detector.forward()
        return det

if __name__ == '__main__':
    main()
