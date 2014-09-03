#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'miyanishi'

import msgpackrpc
import skimage.io
import numpy as np
import snappy

class CaffeClient():
    def __init__(self, hostname="localhost"):
        self.hostname = hostname

    def run_call(self, image):
        client = msgpackrpc.Client(msgpackrpc.Address(self.hostname, 18800))
        feature_dic = client.call('getImageFeatures', snappy.compress(image), image.shape)
        feature_dic = {layer:np.fromstring(snappy.uncompress(features), dtype=np.float32) for layer,features in feature_dic.items()}
        labels = client.call('getImageLabels')
        return feature_dic, labels

    def run_call_async(self, image):
        client = msgpackrpc.Client(msgpackrpc.Address("localhost", 18800))
        feature_dic = client.call_async('getImageFeatures', snappy.compress(image), image.shape)
        labels = client.call('getImageLabels')
        return feature_dic, labels

def load_iamge(image_file_path, color=True):
    image = skimage.img_as_float(skimage.io.imread(image_file_path)).astype(np.float32)
    if image.ndim == 2:
        image = image[:, :, np.newaxis]
        if color:
            image = np.tile(image, (1, 1, 3))
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    return image

if __name__ == '__main__':
    hostname = "localhost"
    image_file_path = "/Users/miyanishi/Data/Source/caffe/examples/images/cat.jpg"
    image = load_iamge(image_file_path)
    client = CaffeClient(hostname)
    featureDic, label = client.run_call(image)
    print featureDic
    print label
