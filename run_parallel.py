#
#  ColorHandPose3DNetwork - Network for estimating 3D Hand Pose from a single RGB Image
#  Copyright (C) 2017  Christian Zimmermann
#  
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
from __future__ import print_function, unicode_literals

import tensorflow as tf
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D
from testcam import WebcamVideoStream
import time

from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d

if __name__ == '__main__':
    # network input
    image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))
    
    # Hand side and evaluation are not required for 2D inference
    #hand_side_tf = tf.constant([[1.0, 0.0]])  # left hand (true for all samples provided)
    #evaluation = tf.placeholder_with_default(True, shape=())

    # build network
    net = ColorHandPose3DNetwork()
    
    #hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, keypoints_scoremap_tf, keypoint_coord3d_tf = net.inference(image_tf, hand_side_tf, evaluation)

    keypoints_scoremap_tf, image_crop_tf, scale_crop_tf, center_tf, raw_scoremap_tf = net.inference2d(image_tf)
    
    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # initialize network
    net.init(sess)
    vs = WebcamVideoStream(src=0).start()

    # Feed list of image paths into pipeline
    #for img_name in image_list:
    while (True):
        
        image_v, image_raw = vs.read()
        
        # Our operations on the frame come here
        #rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Reads in image using scipy module
        #image_raw = scipy.misc.imread(img_name)
        #image_raw = scipy.misc.imresize(rgb, (240, 320))
        
        #image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

        # Full pipeline
        #hand_scoremap_v, image_crop_v, scale_v, center_v, keypoints_scoremap_v, keypoint_coord3d_v = sess.run([hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, keypoints_scoremap_tf, keypoint_coord3d_tf],feed_dict={image_tf: image_v})
        
        start = time.time()
        # 2D pipeline. Run Neural networks and get predictions
        keypoints_scoremap_v, image_crop_v, scale_v, center_v, raw_scoremap_v = sess.run([keypoints_scoremap_tf, image_crop_tf, scale_crop_tf, center_tf, raw_scoremap_tf], feed_dict={image_tf: image_v})
        
        end = time.time()
        
        if (raw_scoremap_v.max() < 0.99995):
            image_bgr = cv2.cvtColor(image_raw, cv2.COLOR_RGB2BGR)
            cv2.imshow('Hand3d', image_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                vs.stop()
                break
            continue
        
        # TODO:
        #--- FILTER FRAMES BY SCORE. IF HAND NOT DETECTED, DON'T RUN POSENET
        
        #print ("Keypoints scores size: ", keypoints_scoremap_v.shape, "\n")
        #print ("Keypoints scores: ", keypoints_scoremap_v, "\n")
        
        #hand_scoremap_v = np.squeeze(hand_scoremap_v)
        image_crop_v = np.squeeze(image_crop_v)
        keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
        #keypoint_coord3d_v = np.squeeze(keypoint_coord3d_v)

        # post processing
        image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
        coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
        
        coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)
        
        plot_hand(coord_hw, image_raw)
        
        # visualize
        #fig = plt.figure(1)
        
        #ax1 = fig.add_subplot(221)
        #ax2 = fig.add_subplot(222)
        #ax3 = fig.add_subplot(223)
        #ax4 = fig.add_subplot(224, projection='3d')
        #ax4 = fig.add_subplot(224)
        
        #ax1.imshow(image_raw)
        #plot_hand(coord_hw, ax1, image_raw)
        
        #ax2.imshow(image_crop_v)
        #plot_hand(coord_hw_crop, ax2, image_crop_v)
        
        #ax3.imshow(image_raw)
        #ax4.imshow(image_crop_v)
        
        #ax3.imshow(np.argmax(hand_scoremap_v, 2))
        #plot_hand_3d(keypoint_coord3d_v, ax4)
        #ax4.view_init(azim=-90.0, elev=-90.0)  # aligns the 3d coord with the camera view
        #ax4.set_xlim([-3, 3])
        #ax4.set_ylim([-3, 1])
        #ax4.set_zlim([-3, 3])
        
        image_raw = cv2.cvtColor(image_raw, cv2.COLOR_RGB2BGR)
        
        print ("Time taken: %.2f seconds" % (end-start))
        
        cv2.imshow('Hand3d', image_raw)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            vs.stop()
            break

    cv2.destroyAllWindows()