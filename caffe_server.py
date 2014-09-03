#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'miyanishi'

import msgpackrpc
import snappy
import numpy as np
from caffe_extractor import CaffeExtractor


class CaffeServer(object):
    def __init__(self, caffe_root, feature_layers, gpu):
        self.extractor = CaffeExtractor(caffe_root=caffe_root, feature_layers=feature_layers, gpu=gpu)

    def getImageFeatures(self, image , image_shape):
        image = np.fromstring(snappy.uncompress(image), dtype=np.float32)
        image.resize(image_shape)
        feature_dic = self.extractor.getImageFeatures(image)
        feature_dic = {layer:snappy.compress(features) for layer,features in feature_dic.items()}
        return feature_dic

    def getImageLabels(self):
        labels = self.extractor.getImageLabels()
        return labels

caffe_root = "/Users/miyanishi/Data/Source/caffe/"
feature_layers = ["fc6"] #, "fc7", "fc8"]
gpu = True
#gpu = False

hostname = "localhost"
port = 18800

server = msgpackrpc.Server(CaffeServer(caffe_root, feature_layers, gpu))
server.listen(msgpackrpc.Address(hostname, port))
print "server: ", hostname, ",", port, " start!"
server.start()
