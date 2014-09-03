#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'miyanishi'

import caffe
import numpy as np


class CaffeExtractor():
    def __init__(self, caffe_root=None, feature_layers=["fc6"], gpu=True):
        self.feature_layers = feature_layers
        MODEL_FILE = caffe_root + 'examples/imagenet/imagenet_deploy.prototxt'
        PRETRAINED = caffe_root + 'examples/imagenet/caffe_reference_imagenet_model'
        MEAN_FILE = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
        self.net = caffe.Classifier(MODEL_FILE, PRETRAINED, mean=np.load(MEAN_FILE),
                                    channel_swap=(2,1,0),
                                    raw_scale=255,
                                    image_dims=(256, 256))
        #self.net.set_phase_test()
        if gpu:
            self.net.set_mode_gpu()
        else:
            self.net.set_mode_cpu()
        imagenet_labels_filename = caffe_root + 'data/ilsvrc12/synset_words.txt'
        self.labels = np.loadtxt(imagenet_labels_filename, str, delimiter='\t')


    def getImageFeatures(self, image):
        score = self.net.predict([image])
        feature_dic = {layer:np.copy(self.net.blobs[layer].data[4][:,0,0]) for layer in self.feature_layers}
        return feature_dic

    def getImageLabels(self):
        top_k = self.net.blobs['prob'].data[4].flatten().argsort()[-1:-6:-1]
        labels = self.labels[top_k].tolist()
        return labels

