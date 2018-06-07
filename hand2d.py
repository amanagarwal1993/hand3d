# Let's build a new network combining handsegnet and posenet!
from __future__ import print_function, unicode_literals

import tensorflow as tf
import os
import pickle

from utils.general import *

ops = NetworkOps

class ColorHandPose3DNetwork(object):
    """ Network performing 3D pose estimation of a human hand from a single color image. """
    def __init__(self):
        self.crop_size = 256
        self.num_kp = 21

    def init(self, session, weight_files=None, exclude_var_list=None):
        """ Initializes weights from pickled python dictionaries.

            Inputs:
                session: tf.Session, Tensorflow session object containing the network graph
                weight_files: list of str, Paths to the pickle files that are used to initialize network weights
                exclude_var_list: list of str, Weights that should not be loaded
        """
        if exclude_var_list is None:
            exclude_var_list = list()

        if weight_files is None:
            weight_files = ['./weights/handsegnet-rhd.pickle', './weights/posenet-rhd-stb-slr-finetuned.pickle']

        # Initialize with weights
        for file_name in weight_files:
            assert os.path.exists(file_name), "File not found."
            with open(file_name, 'rb') as fi:
                weight_dict = pickle.load(fi)
                weight_dict = {k: v for k, v in weight_dict.items() if not any([x in k for x in exclude_var_list])}
                if len(weight_dict) > 0:
                    init_op, init_feed = tf.contrib.framework.assign_from_values(weight_dict)
                    session.run(init_op, init_feed)
                    print('Loaded %d variables from %s' % (len(weight_dict), file_name))

    def inference2d(self, image):
        """ Only 2D part of the pipeline: HandSegNet + PoseNet.

            Inputs:
                image: [B, H, W, 3] tf.float32 tensor, Image with mean subtracted

            Outputs:
                image_crop: [B, 256, 256, 3] tf.float32 tensor, Hand cropped input image
                scale_crop: [B, 1] tf.float32 tensor, Scaling between input image and image_crop
                center: [B, 1] tf.float32 tensor, Center of image_crop wrt to image
                keypoints_scoremap: [B, 256, 256, 21] tf.float32 tensor, Scores for the hand keypoints
                scoremap: [B,240,320] tf.float32 tensor, the detected scoremap output by HandSegnet
        """
        # use network for hand segmentation for detection
        hand_scoremap = self.inference_detection(image)
        hand_scoremap = hand_scoremap[-1]
        
        # Intermediate data processing
        hand_mask, scoremap = single_obj_scoremap(hand_scoremap)
        center, _, crop_size_best = calc_center_bb(hand_mask)
        crop_size_best *= 1.25
        scale_crop = tf.minimum(tf.maximum(self.crop_size / crop_size_best, 0.25), 5.0)
        image_crop = crop_image_from_xy(image, center, self.crop_size, scale=scale_crop)

        # detect keypoints in 2D
        s = image_crop.get_shape().as_list()
        keypoints_scoremap = self.inference_pose2d(image_crop)
        keypoints_scoremap = keypoints_scoremap[-1]
        keypoints_scoremap = tf.image.resize_images(keypoints_scoremap, (s[1], s[2]))
        return keypoints_scoremap, image_crop, scale_crop, center, scoremap

    @staticmethod
    def inference_detection(image, train=False):
        """ HandSegNet: Detects the hand in the input image by segmenting it.

            Inputs:
                image: [B, H, W, 3] tf.float32 tensor, Image with mean subtracted
                train: bool, True in case weights should be trainable

            Outputs:
                scoremap_list_large: list of [B, 256, 256, 2] tf.float32 tensor, Scores for the hand segmentation classes
        """
        with tf.variable_scope('HandSegNet'):
            scoremap_list = list()
            layers_per_block = [2, 2, 4, 4]
            out_chan_list = [64, 128, 256, 512]
            pool_list = [True, True, True, False]

            # learn some feature representation, that describes the image content well
            x = image
            for block_id, (layer_num, chan_num, pool) in enumerate(zip(layers_per_block, out_chan_list, pool_list), 1):
                for layer_id in range(layer_num):
                    x = ops.conv_relu(x, 'conv%d_%d' % (block_id, layer_id+1), kernel_size=3, stride=1, out_chan=chan_num, trainable=train)
                if pool:
                    x = ops.max_pool(x, 'pool%d' % block_id)

            x = ops.conv_relu(x, 'conv5_1', kernel_size=3, stride=1, out_chan=512, trainable=train)
            encoding = ops.conv_relu(x, 'conv5_2', kernel_size=3, stride=1, out_chan=128, trainable=train)

            # use encoding to detect initial scoremap
            x = ops.conv_relu(encoding, 'conv6_1', kernel_size=1, stride=1, out_chan=512, trainable=train)
            scoremap = ops.conv(x, 'conv6_2', kernel_size=1, stride=1, out_chan=2, trainable=train)
            scoremap_list.append(scoremap)

            # upsample to full size
            s = image.get_shape().as_list()
            scoremap_list_large = [tf.image.resize_images(x, (s[1], s[2])) for x in scoremap_list]

        return scoremap_list_large

    def inference_pose2d(self, image_crop, train=False):
        """ PoseNet: Given an image it detects the 2D hand keypoints.
            The image should already contain a rather tightly cropped hand.

            Inputs:
                image: [B, H, W, 3] tf.float32 tensor, Image with mean subtracted
                train: bool, True in case weights should be trainable

            Outputs:
                scoremap_list_large: list of [B, 256, 256, 21] tf.float32 tensor, Scores for the hand keypoints
        """
        with tf.variable_scope('PoseNet2D'):
            scoremap_list = list()
            layers_per_block = [2, 2, 4, 2]
            out_chan_list = [64, 128, 256, 512]
            pool_list = [True, True, True, False]

            # learn some feature representation, that describes the image content well
            x = image_crop
            for block_id, (layer_num, chan_num, pool) in enumerate(zip(layers_per_block, out_chan_list, pool_list), 1):
                for layer_id in range(layer_num):
                    x = ops.conv_relu(x, 'conv%d_%d' % (block_id, layer_id+1), kernel_size=3, stride=1, out_chan=chan_num, trainable=train)
                if pool:
                    x = ops.max_pool(x, 'pool%d' % block_id)

            x = ops.conv_relu(x, 'conv4_3', kernel_size=3, stride=1, out_chan=256, trainable=train)
            x = ops.conv_relu(x, 'conv4_4', kernel_size=3, stride=1, out_chan=256, trainable=train)
            x = ops.conv_relu(x, 'conv4_5', kernel_size=3, stride=1, out_chan=256, trainable=train)
            x = ops.conv_relu(x, 'conv4_6', kernel_size=3, stride=1, out_chan=256, trainable=train)
            encoding = ops.conv_relu(x, 'conv4_7', kernel_size=3, stride=1, out_chan=128, trainable=train)

            # use encoding to detect initial scoremap
            x = ops.conv_relu(encoding, 'conv5_1', kernel_size=1, stride=1, out_chan=512, trainable=train)
            scoremap = ops.conv(x, 'conv5_2', kernel_size=1, stride=1, out_chan=self.num_kp, trainable=train)
            scoremap_list.append(scoremap)

            # iterate recurrent part a couple of times
            layers_per_recurrent_unit = 5
            num_recurrent_units = 2
            for pass_id in range(num_recurrent_units):
                x = tf.concat([scoremap_list[-1], encoding], 3)
                for rec_id in range(layers_per_recurrent_unit):
                    x = ops.conv_relu(x, 'conv%d_%d' % (pass_id+6, rec_id+1), kernel_size=7, stride=1, out_chan=128, trainable=train)
                x = ops.conv_relu(x, 'conv%d_6' % (pass_id+6), kernel_size=1, stride=1, out_chan=128, trainable=train)
                scoremap = ops.conv(x, 'conv%d_7' % (pass_id+6), kernel_size=1, stride=1, out_chan=self.num_kp, trainable=train)
                scoremap_list.append(scoremap)

            scoremap_list_large = scoremap_list

        return scoremap_list_large