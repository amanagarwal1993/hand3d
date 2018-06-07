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
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d

if __name__ == '__main__':
    # images to be shown
    image_list = list()
    image_list.append('./data/img.png')
    image_list.append('./data/img2.png')
    image_list.append('./data/img3.png')
    image_list.append('./data/img4.png')
    image_list.append('./data/img5.png')
    image_list.append('./data/1.png')
    image_list.append('./data/2.png')
    image_list.append('./data/3.png')
    image_list.append('./data/4.png')
    image_list.append('./data/5.png')
    image_list.append('./data/6.png')

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

    # Feed list of image paths into pipeline
    for img_name in image_list:
        # Reads in image using scipy module
        image_raw = scipy.misc.imread(img_name)
        image_raw = scipy.misc.imresize(image_raw, (240, 320))
        image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

        # Full pipeline
        #hand_scoremap_v, image_crop_v, scale_v, center_v, keypoints_scoremap_v, keypoint_coord3d_v = sess.run([hand_scoremap_tf, image_crop_tf, scale_tf, center_tf, keypoints_scoremap_tf, keypoint_coord3d_tf],feed_dict={image_tf: image_v})
        
        # 2D pipeline (Handseg + Posenet)
        keypoints_scoremap_v, image_crop_v, scale_v, center_v, raw_scoremap_v = sess.run([keypoints_scoremap_tf, image_crop_tf, scale_crop_tf, center_tf, raw_scoremap_tf], feed_dict={image_tf: image_v})

        if (raw_scoremap_v.max() < 0.99999):
            continue
        
        #hand_scoremap_v = np.squeeze(hand_scoremap_v)
        image_crop_v = np.squeeze(image_crop_v)
        keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)
        
        # post processing
        image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
        coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))
        
        #--- Uncomment these lines
        #coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)

        #plot_hand(coord_hw, image_raw)
        
        #image_raw = cv2.cvtColor(image_raw, cv2.COLOR_RGB2BGR)
        
        #plt.imshow(image_raw)
        #plt.show()
