import argparse
import caffe
import csv
import logging
import numpy as np
import os
import sys

cur_dir = os.path.abspath(os.path.dirname(__file__))


class SceneClassifier():

    def __init__(self,
                 prototxt_file=os.path.join(cur_dir, 'resources', 'deploy_resnet152_places365.prototxt'),
                 caffemodel_file=os.path.join(cur_dir, 'resources', 'resnet152_places365.caffemodel'),
                 scene_hierarchy_file=os.path.join(cur_dir, 'resources', 'scene_hierarchy_places365.csv'),
                 use_gpu=True):

        # read scene_hierarchy file to get lvl1 meta information
        hierarchy_places3 = []
        with open(scene_hierarchy_file, 'r') as csvfile:
            content = csv.reader(csvfile, delimiter=',')
            next(content)  # skip explanation line
            next(content)  # skip explanation line
            for line in content:
                hierarchy_places3.append(line[1:4])

        hierarchy_places3 = np.asarray(hierarchy_places3, dtype=np.float)

        # normalize label if it belongs to multiple categories
        self.hierarchy_places3 = hierarchy_places3 / np.expand_dims(np.sum(hierarchy_places3, axis=1), axis=-1)

        # initialize network
        if use_gpu:
            caffe.set_mode_cpu()  # CPU
        else:
            caffe.set_mode_gpu()  # GPU

        caffe.set_device(0)

        self.net = caffe.Net(prototxt_file, caffemodel_file, caffe.TEST)

        # steps for image preprocessing
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2, 0, 1))
        self.transformer.set_channel_swap('data', (2, 1, 0))
        self.transformer.set_raw_scale('data', 255.0)

        # define network input dimensions
        self.net.blobs['data'].reshape(1, 3, 224, 224)

    def get_scene_probabilities(self, img_path):
        # load and preprocess image
        img = caffe.io.load_image(img_path)
        img = self.transformer.preprocess('data', img)

        # feed image data into cnn
        self.net.blobs['data'].data[...] = [img]
        scene_probs = self.net.forward()['prob']

        # get the probabilites of the lvl 1 categories
        places3_prob = np.matmul(scene_probs, self.hierarchy_places3)[0]

        logging.info('indoor : {}'.format(places3_prob[0]))
        logging.info('natural: {}'.format(places3_prob[1]))
        logging.info('urban  : {}'.format(places3_prob[2]))

        return places3_prob

    def get_scene_label(self, scene_prob):
        scene_label = np.argmax(scene_prob, axis=0)

        if places3_label == 0:
            logging.info('Images shows indoor scenery!')
        elif places3_label == 1:
            logging.info('Images shows natural scenery!')
        elif places3_label == 2:
            logging.info('Images shows urban scenery!')

        return scene_label


'''
########################################################################################################################
# MAIN TO TEST CLASS
########################################################################################################################
'''


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
    parser.add_argument('-i', '--image', type=str, required=True, help='path to image file')
    args = parser.parse_args()
    return args


def main():
    # load arguments
    args = parse_args()

    # define logging level and format
    level = logging.ERROR
    if args.verbose:
        level = logging.DEBUG
    logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=level)

    # init scene classifier
    sc = SceneClassifier(use_gpu=True)

    # predict scene label
    places3_prob = sc.get_scene_probabilities(args.image)
    places3_label = sc.get_scene_label(places3_prob)

    return 0


if __name__ == '__main__':
    sys.exit(main())