import argparse
import caffe
import csv
import json
import numpy as np
import os
import re
import s2sphere as s2
import sys
from scipy.misc import imresize as imresize
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses unnecessarily excessive console output
import tensorflow as tf
import tensorflow.contrib.slim as slim

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))
import cnn_architectures


class GeoEstimator():

    def __init__(self, model_file, cnn_input_size=224, scope=None, use_cpu=False):
        print('Initialize {} geolocation model.'.format(scope))

        self._cnn_input_size = cnn_input_size

        # load model config
        with open(os.path.join(os.path.dirname(model_file), 'cfg.json')) as cfg_file:
            cfg = json.load(cfg_file)

        # get partitioning
        print('\tGet geographical partitioning(s) ... ')
        partitioning_files = []
        for partitioning in cfg['partitionings']:
            partitioning_files.append(os.path.join(os.path.dirname(__file__), 'geo-cells', partitioning))

        self._num_partitionings = len(partitioning_files)

        # red geo partitioning
        classes_geo, hexids2classes, class2hexid, cell_centers = self._read_partitioning(partitioning_files)

        self._classes_geo = classes_geo
        self._cell_centers = cell_centers

        # get geographical hierarchy
        self._cell_hierarchy = self._get_geographical_hierarchy(classes_geo, hexids2classes, class2hexid, cell_centers)

        # build cnn
        self._image_ph = tf.placeholder(shape=[3, self._cnn_input_size, self._cnn_input_size, 3], dtype=tf.float32)

        config = tf.ConfigProto()
        #config.log_device_placement = True
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        print('\tRestore model from: {}'.format(model_file))

        if scope is not None:
            with tf.variable_scope(scope) as scope:
                self.scope = scope
        else:
            self.scope = tf.get_variable_scope()

        if use_cpu:
            device = '/cpu:0'
        else:
            device = '/gpu:0'

        with tf.variable_scope(self.scope):
            with tf.device(device):
                net, _ = cnn_architectures.create_model(
                    cfg['architecture'], self._image_ph, is_training=False, num_classes=None, reuse=None)

                with tf.variable_scope('classifier_geo', reuse=None):
                    self.logits = slim.conv2d(
                        net, np.sum(classes_geo), [1, 1], activation_fn=None, normalizer_fn=None, scope='logits')
                    self.logits = tf.squeeze(self.logits)

        var_list = {
            re.sub('^' + self.scope.name + '/', '', x.name)[:-2]: x for x in tf.global_variables(self.scope.name)
        }
        saver = tf.train.Saver(var_list=var_list)

        saver.restore(self.sess, model_file)

        # get activations from last conv layer and output weights in order to calculate class activation maps
        self.activations = tf.get_default_graph().get_tensor_by_name(self.scope.name + '_1/resnet_v2_101/activations:0')
        self.activation_weights = tf.get_default_graph().get_tensor_by_name(self.scope.name +
                                                                            '/classifier_geo/logits/weights:0')

    def get_prediction(self, img_path, show_cam=True):
        # read image
        img_path_ph = tf.placeholder(shape=[], dtype=tf.string)
        img_content = tf.read_file(img_path_ph)

        # decode image
        img = tf.image.decode_jpeg(img_content, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)

        # normalize image to -1 .. 1
        img = tf.subtract(img, 0.5)
        img = tf.multiply(img, 2.0)

        # crop image into three evenly sampled crops
        img, img_crops, bboxes = self._crop_image(img)

        # apply image transformations
        img_v, img_crops_v, bboxes_v = self.sess.run([img, img_crops, bboxes], feed_dict={img_path_ph: img_path})

        # feed forward batch of images in cnn and extract result
        activations_v, activation_weights_v, logits_v = self.sess.run(
            [self.activations, self.activation_weights, self.logits], feed_dict={self._image_ph: img_crops_v})

        # softmax to get class probabilities with sum 1
        logits_v[0, :] = self._softmax(logits_v[0, :])
        logits_v[1, :] = self._softmax(logits_v[1, :])
        logits_v[2, :] = self._softmax(logits_v[2, :])

        # fuse results of image crops using the maximum
        logits_v = np.max(logits_v, axis=0)

        # assign logits to respective partitionings and get prediction (class with highest probability)
        partitioning_logits = []
        partitioning_pred = []
        for p in range(self._num_partitionings):
            p_logits = logits_v[np.sum(self._classes_geo[0:p + 1]):np.sum(self._classes_geo[0:p + 2])]
            p_pred = p_logits.argsort()
            partitioning_logits.append(p_logits)
            partitioning_pred.append(p_pred[-1])

        # get hierarchical multipartitioning results
        hierarchical_logits = partitioning_logits[-1]  # get logits from finest partitioning
        if self._num_partitionings > 1:
            for c in range(self._classes_geo[-1]):  # num_logits of finest partitioning
                for p in range(self._num_partitionings - 1):
                    hierarchical_logits[c] *= partitioning_logits[p][self._cell_hierarchy[c][p]]

            pred = hierarchical_logits.argsort()
            partitioning_pred.append(pred[-1])

        predicted_cell_id = partitioning_pred[-1]

        # get gps coordinate from class
        lat, lng = self._cell_centers[self._num_partitionings - 1][predicted_cell_id]
        print('Predicted cell id: {}'.format(predicted_cell_id))
        print('Predicted coordinate (lat, lng): ({}, {})'.format(lat, lng))

        # get class activation map of hierarchical prediction in the finest partitioning
        # NOTE: prediction is the id in the finest partitioning, Pay attention to offset due to coarser partitionings
        # Cropping the activations_weights and logits to solve the issue (   np.sum(self._classes_geo[0:p + 1]):   )
        p = self._num_partitionings - 1
        activation_weights_v = activation_weights_v[:, :, :, np.sum(self._classes_geo[0:p + 1]):]
        logits_v = logits_v[np.sum(self._classes_geo[0:p + 1]):]

        print(activation_weights_v.shape)

        cam = self.get_class_activation_map(activations_v, activation_weights_v, bboxes_v, predicted_cell_id,
                                            [img_v.shape[0], img_v.shape[1]])

        if show_cam:
            img_ovlr = self.create_cam_heatmap(img_v, cam)
            plt.imshow(img_ovlr)
            plt.show()

        return {
            'predicted_cell_id': predicted_cell_id,
            'lat': lat,
            'lng': lng,
            'cam': cam,
            'activations': activations_v,
            'activation_weights': activation_weights_v,
            'logits': logits_v
        }

    def create_cam_heatmap(self, img, cam, img_alpha=0.6):
        # create rgb overlay
        cm = plt.get_cmap('jet')
        cam_ovlr = cm(cam)

        # normalize to 0..1 and convert to grayscale
        img = (img + 1) / 2
        img_gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]

        # create heatmap composite
        return img_alpha * np.expand_dims(img_gray, axis=-1) + (1 - img_alpha) * cam_ovlr[:, :, 0:3]

    def get_class_activation_map(self, activations, activation_weights, bboxes, class_idx, output_size):
        # get weights of specified class
        prediction_activation_weights = activation_weights[0, 0, :, class_idx]

        # get dimensions
        num_crops, h, w, num_features = activations.shape
        img_size = bboxes[0][-1]

        r = h / img_size

        # create output variables
        cam = np.zeros(shape=[int(bboxes[-1][0] * r + 0.5) + w, int(bboxes[-1][1] * r + 0.5) + h])
        num_activations = np.zeros(shape=[int(bboxes[-1][0] * r + 0.5) + w, int(bboxes[-1][1] * r + 0.5) + h])

        for crop_idx in range(num_crops):
            # get activation map of current crop
            crop_activations = activations[crop_idx, :, :, :]
            crop_activation_map = prediction_activation_weights.dot(crop_activations.reshape((num_features, h * w)))
            crop_activation_map = crop_activation_map.reshape(h, w)

            # save values
            feature_bbox = []
            for entry in bboxes[crop_idx]:
                feature_bbox.append(int(entry * r + 0.5))

            cam[feature_bbox[0]:feature_bbox[0] + w, feature_bbox[1]:feature_bbox[1] + h] += crop_activation_map
            num_activations[feature_bbox[0]:feature_bbox[0] + w, feature_bbox[1]:feature_bbox[1] + h] += 1

        # NOTE: prevent division by 0, if the whole image is not covered with all crops [max_dim > 3 * min_dim]
        num_activations[num_activations == 0] = 1

        # normalize class activation map
        cam /= num_activations
        cam -= np.min(cam)
        cam /= np.max(cam)
        cam = np.asarray(cam * 255 + 0.5, dtype=np.uint8)

        cam = imresize(cam, output_size)

        return cam

    def _softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def _crop_image(self, img):
        height = tf.to_float(tf.shape(img)[0])
        width = tf.to_float(tf.shape(img)[1])

        # get minimum and maximum coordinate
        max_side_len = tf.maximum(width, height)
        min_side_len = tf.minimum(width, height)
        is_w, is_h = tf.cond(tf.less(width, height), lambda: (0, 1), lambda: (1, 0))

        # resize image
        ratio = self._cnn_input_size / min_side_len
        offset = (tf.to_int32(max_side_len * ratio + 0.5) - self._cnn_input_size) // 2
        img = tf.image.resize_images(img, size=[tf.to_int32(height * ratio + 0.5), tf.to_int32(width * ratio + 0.5)])

        # get crops according to image orientation
        img_array = []
        bboxes = []
        for i in range(3):
            bbox = [
                i * is_h * offset, i * is_w * offset,
                tf.convert_to_tensor(self._cnn_input_size),
                tf.convert_to_tensor(self._cnn_input_size)
            ]
            img_crop = tf.image.crop_to_bounding_box(img, bbox[0], bbox[1], bbox[2], bbox[3])
            img_crop = tf.expand_dims(img_crop, 0)

            bboxes.append(bbox)
            img_array.append(img_crop)

        return img, tf.concat(img_array, axis=0), bboxes

    def _read_partitioning(self, partitioning_files):
        # define vars
        cell_centers = []  # list of cell centers for each class
        classes_geo = []  # list of number of geo_classes for each partitioning
        hexids2classes = []  # dictionary to convert a hexid into a class label
        class2hexid = []  # hexid for each class
        classes_geo.append(0)  # add zero to classes vector for simplification in further processing steps

        # get cell partitionings
        for partitioning in partitioning_files:
            partitioning_hexids2classes = {}
            partitioning_classes_geo = 0
            partitioning_class2hexid = []
            partitioning_cell_centers = []

            with open(partitioning, 'r') as cell_file:
                cell_reader = csv.reader(cell_file, delimiter=',')
                for line in cell_reader:
                    if len(line) > 1 and line[0] not in ['num_images', 'min_concept_probability', 'class_label']:
                        partitioning_hexids2classes[line[1]] = int(line[0])
                        partitioning_class2hexid.append(line[1])
                        partitioning_cell_centers.append([float(line[3]), float(line[4])])
                        partitioning_classes_geo += 1

                hexids2classes.append(partitioning_hexids2classes)
                class2hexid.append(partitioning_class2hexid)
                cell_centers.append(partitioning_cell_centers)
                classes_geo.append(partitioning_classes_geo)

        return classes_geo, hexids2classes, class2hexid, cell_centers

    # generate hierarchical list of respective higher-order geo cells for multipartitioning [|c| x |p|]
    def _get_geographical_hierarchy(self, classes_geo, hexids2classes, class2hexid, cell_centers):
        cell_hierarchy = []

        if self._num_partitionings > 1:
            # loop through finest partitioning
            for c in range(classes_geo[-1]):
                cell_bin = self._hextobin(class2hexid[-1][c])
                level = int(len(cell_bin[3:-1]) / 2)
                parents = []

                # get parent cells
                for l in reversed(range(2, level + 1)):
                    hexid_parent = self._create_cell(cell_centers[-1][c][0], cell_centers[-1][c][1], l)
                    for p in reversed(range(self._num_partitionings - 1)):
                        if hexid_parent in hexids2classes[p]:
                            parents.append(hexids2classes[p][hexid_parent])

                    if len(parents) == self._num_partitionings - 1:
                        break

                cell_hierarchy.append(parents[::-1])

        return cell_hierarchy

    def _hextobin(self, hexval):
        thelen = len(hexval) * 4
        binval = bin(int(hexval, 16))[2:]
        while ((len(binval)) < thelen):
            binval = '0' + binval

        binval = binval.rstrip('0')
        return binval

    def _hexid2latlng(self, hexid):
        # convert hexid to latlng of cellcenter
        cellid = s2.CellId().from_token(hexid)
        cell = s2.Cell(cellid)
        point = cell.get_center()
        latlng = s2.LatLng(0, 0).from_point(point).__repr__()
        _, latlng = latlng.split(' ', 1)
        lat, lng = latlng.split(',', 1)
        lat = float(lat)
        lng = float(lng)
        return lat, lng

    def _latlng2class(self, lat, lng, hexids2classes):
        for l in range(2, 18):  # NOTE: upper boundary necessary
            hexid = create_cell(lat, lng, l)
            if hexid in hexids2classes:
                return hexids2classes[hexid]

    def _create_cell(self, lat, lng, level):
        p1 = s2.LatLng.from_degrees(lat, lng)
        cell = s2.Cell.from_lat_lng(p1)
        cell_parent = cell.id().parent(level)
        hexid = cell_parent.to_token()
        return hexid


'''
########################################################################################################################
# MAIN TO TEST CLASS
########################################################################################################################
'''


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--image', type=str, required=True, help='path to image file')
    parser.add_argument('-m', '--model', type=str, required=True, help='path to model file')
    parser.add_argument('-c', '--cpu', action='store_true', help='use cpu')
    args = parser.parse_args()
    return args


def main():
    # load arguments
    args = parse_args()

    # check if gpu is available
    if not tf.test.is_gpu_available():
        print('No GPU available. Using CPU instead ... ')
        args.cpu = True

    # init scene classifier
    ge = GeoEstimator(args.model, use_cpu=args.cpu)

    # predict scene label
    pred = ge.get_prediction(args.image)

    return 0


if __name__ == '__main__':
    sys.exit(main())
