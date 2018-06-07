import cv2
import scipy.misc
import numpy as np
from threading import Thread
from skimage.transform import resize
import time

class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, wait and read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        image_raw = scipy.misc.imresize(frame, (240, 320))
        image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)
        return image_v, image_raw

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.stream.release()
