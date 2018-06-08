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

from __future__ import print_function, unicode_literals

import tensorflow as tf
import numpy as np
import scipy.misc
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import strftime, gmtime

from nets.ColorHandPose3DNetwork import ColorHandPose3DNetwork
from utils.general import detect_keypoints, trafo_coords, plot_hand, plot_hand_3d

def detect(img_name):
    # images to be shown
    
    # network input
    image_tf = tf.placeholder(tf.float32, shape=(1, 240, 320, 3))
    
    # build network
    net = ColorHandPose3DNetwork()
    
    keypoints_scoremap_tf, image_crop_tf, scale_crop_tf, center_tf, raw_scoremap_tf = net.inference2d(image_tf)
    
    # Start TF
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # initialize network
    net.init(sess)

    # Reads in image using scipy module
    image_raw = scipy.misc.imread(img_name)
    image_raw = scipy.misc.imresize(image_raw, (240, 320))
    image_v = np.expand_dims((image_raw.astype('float') / 255.0) - 0.5, 0)

    # 2D pipeline (Handseg + Posenet)
    keypoints_scoremap_v, image_crop_v, scale_v, center_v, raw_scoremap_v = sess.run([keypoints_scoremap_tf, image_crop_tf, scale_crop_tf, center_tf, raw_scoremap_tf], feed_dict={image_tf: image_v})

    if (raw_scoremap_v.max() < 0.99999):
        continue

    image_crop_v = np.squeeze(image_crop_v)
    keypoints_scoremap_v = np.squeeze(keypoints_scoremap_v)

    # post processing
    image_crop_v = ((image_crop_v + 0.5) * 255).astype('uint8')
    coord_hw_crop = detect_keypoints(np.squeeze(keypoints_scoremap_v))

    coord_hw = trafo_coords(coord_hw_crop, center_v, scale_v, 256)

    plot_hand(coord_hw, image_raw)

    #image_raw = cv2.cvtColor(image_raw, cv2.COLOR_RGB2BGR)

    plt.imshow(image_raw)
    plt.savefig('result/detected_image.png')
    