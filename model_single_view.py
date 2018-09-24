from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy import misc
from bilinear_sampler import bilinear_sampler
import math
#from tf.contrib.resampler import resampler

class Net(object):
    # def initalize(self, sess):
    #     pre_trained_weights = np.load(open(self.weight_path, "rb"), encoding="latin1").item()
    #     keys = sorted(pre_trained_weights.keys())
    #     #for k in keys:
    #     for k in list(filter(lambda x: 'conv' in x,keys)):
    #         with tf.variable_scope(k, reuse=True):
    #             temp = tf.get_variable('weights')
    #             sess.run(temp.assign(pre_trained_weights[k]['weights']))
    #         with tf.variable_scope(k, reuse=True):
    #             temp = tf.get_variable('biases')
    #             sess.run(temp.assign(pre_trained_weights[k]['biases']))

    # def conv(self, input_, filter_size, in_channels, out_channels, name, strides, padding, groups, pad_input=1):
    #     if pad_input==1:
    #         paddings = tf.constant([ [0, 0], [1, 1,], [1, 1], [0, 0] ])
    #         input_ = tf.pad(input_, paddings, "CONSTANT")

    #     with tf.variable_scope(name) as scope:
    #         filt = tf.get_variable('weights', shape=[filter_size, filter_size, int(in_channels/groups), out_channels], trainable=self.trainable)
    #         bias = tf.get_variable('biases',  shape=[out_channels], trainable=self.trainable)
    #     if groups == 1:
    #         return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_, filt, strides=strides, padding=padding), bias))
    #     else:
    #         # Split input_ and weights and convolve them separately
    #         input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=input_)
    #         filt_groups = tf.split(axis = 3, num_or_size_splits=groups, value=filt)
    #         output_groups = [ tf.nn.conv2d( i, k, strides = strides, padding = padding) for i,k in zip(input_groups, filt_groups)]

    #         conv = tf.concat(axis = 3, values = output_groups)
    #         return tf.nn.relu(tf.nn.bias_add(conv, bias))

    def conv(self, input_, filter_size, in_channels, out_channels, name, strides, padding, groups, pad_input=1, relu=1, pad_num=1):
        if pad_input==1:
            paddings = tf.constant([ [0, 0], [pad_num, pad_num,], [pad_num, pad_num], [0, 0] ])
            input_ = tf.pad(input_, paddings, "CONSTANT")

        with tf.variable_scope(name) as scope:
            filt = tf.get_variable('weights', shape=[filter_size, filter_size, int(in_channels/groups), out_channels], trainable=self.trainable)
            bias = tf.get_variable('biases',  shape=[out_channels], trainable=self.trainable)
        if groups == 1:
            if relu:
                return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(input_, filt, strides=strides, padding=padding), bias))
            else:
                return tf.nn.bias_add(tf.nn.conv2d(input_, filt, strides=strides, padding=padding), bias)

        else:
            # Split input_ and weights and convolve them separately
            input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=input_)
            filt_groups = tf.split(axis = 3, num_or_size_splits=groups, value=filt)
            output_groups = [ tf.nn.conv2d( i, k, strides = strides, padding = padding) for i,k in zip(input_groups, filt_groups)]

            conv = tf.concat(axis = 3, values = output_groups)
            if relu:
                return tf.nn.relu(tf.nn.bias_add(conv, bias))
            else:
                return tf.nn.bias_add(conv, bias)

    def fc(self, input_, in_channels, out_channels, name, relu):
        input_ = tf.reshape(input_ , [-1, in_channels])
        with tf.variable_scope(name) as scope:
            filt = tf.get_variable('weights', shape=[in_channels , out_channels], trainable=self.trainable)
            bias = tf.get_variable('biases',  shape=[out_channels], trainable=self.trainable)
        if relu:
            return tf.nn.relu(tf.nn.bias_add(tf.matmul(input_, filt), bias))
        else:
            return tf.nn.bias_add(tf.matmul(input_, filt), bias)

    # def pool(self, input_, padding, name):
    #     return tf.nn.max_pool(input_, ksize=[1,3,3,1], strides=[1,2,2,1], padding=padding, name= name)

    def model(self):

        debug=True
        net_layers={}
        #placeholder for a random set of <batch_size> images of fixed size -- 224,224
        # self.input_batch_size = tf.shape(self.input_imgs)[0]  # Returns a scalar `tf.Tensor`
        
        self.tform = tf.placeholder(tf.float32, shape = [None, 19], name = "tform")
        net_layers['input_stack'] = self.input_imgs
        #mean is already subtracted in helper.py as part of preprocessing
        # Conv-Layers

        net_layers['Convolution1'] = self.conv(net_layers['input_stack'], 3, 3 , 16, name= 'Convolution1', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        
        net_layers['Convolution2'] = self.conv(net_layers['Convolution1'], 3, 16, 32, name= 'Convolution2', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        
        net_layers['Convolution3'] = self.conv(net_layers['Convolution2'], 3, 32 , 64, name= 'Convolution3', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        
        net_layers['Convolution4'] = self.conv(net_layers['Convolution3'], 3, 64 , 128, name= 'Convolution4', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        
        net_layers['Convolution5'] = self.conv(net_layers['Convolution4'], 3, 128 , 256, name= 'Convolution5', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        
        net_layers['Convolution6'] = self.conv(net_layers['Convolution5'], 3, 256 , 512, name= 'Convolution6', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1)
        
        # print(net_layers['Convolution1'])
        # print(net_layers['Convolution2'])
        # print(net_layers['Convolution3'])
        # print(net_layers['Convolution4'])
        # print(net_layers['Convolution5'])
        # print(net_layers['Convolution6'])
        #fully connected Layer
        net_layers['src_fc6'] = self.fc(net_layers['Convolution6'], 4*4*512 , 4096, name='src_fc6', relu = 1)
        #print(net_layers['src_fc6'])
        #viewpoint transformation
        net_layers['view_fc1'] = self.fc(self.tform, 19 , 128, name='view_fc1', relu = 1)
        net_layers['view_fc2'] = self.fc(net_layers['view_fc1'], 128 , 256, name='view_fc2', relu = 1)
        
        
        ##concatenation
        net_layers['view_concat'] = tf.concat([net_layers['src_fc6'], net_layers['view_fc2']], 1)
        #print(net_layers['view_concat'])
        ##Fully connected
        net_layers['de_fc1'] = self.fc(net_layers['view_concat'], 4352 , 4096, name='de_fc1', relu = 1)
        #print(net_layers['de_fc1'])
        
        net_layers['de_fc2'] = self.fc(net_layers['de_fc1'], 4096 , 4096, name='de_fc2', relu = 1)
        #print(net_layers['de_fc2'])
        net_layers['de_fc3'] = self.fc(net_layers['de_fc2'], 4096 , 8*8*64, name='de_fc3', relu = 1)
        #print(net_layers['de_fc3'])
        net_layers['de_fc3_rs'] = tf.reshape(net_layers['de_fc3'],shape=[-1, 8, 8, 64], name='de_fc3_rs')
        #print(net_layers['de_fc3_rs'])

        #deconv

        deconv1_x2 = tf.image.resize_bilinear(net_layers['de_fc3_rs'], [15, 15])
        net_layers['deconv1'] = self.conv(deconv1_x2, 3, 64 , 256, name= 'deconv1', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1)                                   

        deconv2_x2 = tf.image.resize_bilinear(net_layers['deconv1'], [29, 29])
        net_layers['deconv2'] = self.conv(deconv2_x2, 3, 256 , 128, name= 'deconv2', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1)

        deconv3_x2 = tf.image.resize_bilinear(net_layers['deconv2'], [57, 57])
        net_layers['deconv3'] = self.conv(deconv3_x2, 3, 128 , 64, name= 'deconv3', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1)
        
        deconv4_x2 = tf.image.resize_bilinear(net_layers['deconv3'], [113, 113])
        # net_layers['deconv5'] = tf.nn.tanh(self.conv(deconv5_x2, 3, 32 , 16, name= 'deconv5', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1))
        net_layers['deconv4'] = self.conv(deconv4_x2, 3, 64 , 32, name= 'deconv4', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1)
        
        deconv5_x2 = tf.image.resize_bilinear(net_layers['deconv4'], [225, 225])
        # net_layers['deconv6'] = tf.nn.tanh(self.conv(deconv6_x2, 3, 16 , 2, name= 'deconv6', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1))
        net_layers['deconv5'] = self.conv(deconv5_x2, 3, 32 , 16, name= 'deconv5', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1)

        deconv6_x2 = tf.image.resize_bilinear(net_layers['deconv5'], [224, 224])
        # net_layers['deconv6'] = tf.nn.tanh(self.conv(deconv6_x2, 3, 16 , 2, name= 'deconv6', strides=[1,2,2,1] ,padding='VALID', groups=1,pad_input=1))
        net_layers['deconv6'] = tf.nn.tanh(self.conv(deconv6_x2, 3, 16 , 2, name= 'deconv6', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1))

        # net_layers['deconv1'] = self._upscore_layer(net_layers['de_fc3_rs'], shape=None,
        #                                    num_classes=128,debug=debug, name='deconv1', ksize=3, stride=2, pad_input=1)
        
        # net_layers['deconv2'] = self._upscore_layer(net_layers['deconv1'], shape=None,
        #                                    num_classes=64,debug=debug, name='deconv2', ksize=3, stride=2, pad_input=1)

        # net_layers['deconv3'] = self._upscore_layer(net_layers['deconv2'], shape=None,
        #                                    num_classes=32,debug=debug, name='deconv3', ksize=3, stride=2, pad_input=1)

        # net_layers['deconv4'] = self._upscore_layer(net_layers['deconv3'], shape=None,
        #                                    num_classes=16,debug=debug, name='deconv4', ksize=3, stride=2, pad_input=1)

        # net_layers['deconv5'] = self._upscore_layer(net_layers['deconv4'], shape=None,
        #                                    num_classes=2,debug=debug, name='deconv5', ksize=3, stride=1, pad_input=1)
        

        # deconv1_x2 = tf.image.resize_bilinear(net_layers['de_fc3_rs'], [8, 8])
        # net_layers['deconv1'] = self.conv(deconv1_x2, 3, 64 , 256, name= 'deconv1', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1)

        print(net_layers['deconv1'])
        print(net_layers['deconv2'])
        print(net_layers['deconv3'])
        print(net_layers['deconv4'])
        print(net_layers['deconv5'])
        print(net_layers['deconv6'])
        ##function will handle steps of resizing and adding
        #remap using bilinear on (flow(deconv6) and input_imgs) to get predImg
        

        net_layers['predImg']=bilinear_sampler(self.input_imgs,net_layers['deconv6'], resize=True)
        
        deconv_x2_mask = tf.image.resize_bilinear(net_layers['deconv5'], [224, 224])
        net_layers['deconv_mask'] = self.conv(deconv_x2_mask, 3, 16 , 2, name= 'deconv_mask', strides=[1,1,1,1] ,padding='VALID', groups=1,pad_input=1)


        # print(net_layers['deconv6'])
        #add coords
        #resize
        #call tf resampler
        
        self.net_layers = net_layers



    def _upscore_layer(self, bottom, shape,num_classes, name, debug, ksize=3, stride=2, pad_input=1, relu=1):

        strides = [1, stride, stride, 1]
        with tf.variable_scope(name):
            in_features = bottom.get_shape()[3].value
            print(bottom)
            if shape is None:
                # Compute shape out of Bottom
                in_shape = bottom.get_shape()
                h = ((in_shape[1].value - 1) * stride) + 1
                w = ((in_shape[2].value - 1) * stride) + 1
                new_shape = [in_shape[0].value, h, w, num_classes]
            else:
                new_shape = [shape[0], shape[1], shape[2], num_classes]


            deconv_shape = tf.stack([self.batch_size, new_shape[1], new_shape[2], num_classes])


            #logging.debug("Layer: %s, Fan-in: %d" % (name, in_features))
            f_shape = [ksize, ksize, num_classes, in_features]
            # create
            num_input = ksize * ksize * in_features / stride
            stddev = (2 / num_input)**0.5

            ##add padding
            if pad_input==1:
                paddings = tf.constant([ [0, 0], [1, 1,], [1, 1], [0, 0] ])
                #bottom = tf.pad(bottom, paddings, "CONSTANT")
            weights = self.get_deconv_filter(f_shape)
            if relu==1:
                deconv = tf.nn.relu(tf.nn.conv2d_transpose(bottom, weights, deconv_shape,
                                            strides=strides, padding='SAME'))
            else:
                deconv = tf.nn.conv2d_transpose(bottom, weights, deconv_shape,
                                            strides=strides, padding='SAME')

            if debug:
                deconv = tf.Print(deconv, [tf.shape(deconv)],
                                  message='Shape of %s' % name,
                                  summarize=4, first_n=1)


        return deconv

    def get_deconv_filter(self, f_shape):
        width = f_shape[0]
        height = f_shape[1]
        f = math.ceil(width/2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="up_filter", initializer=init,
                               shape=weights.shape)

    def reconstruction_loss(self,real_images, generated_images, mask):
        """
        The reconstruction loss is defined as the sum of the L1 distances
        between the target images and their generated counterparts
        """
        curr_exp = tf.nn.softmax(mask)
        curr_proj_error = tf.abs(real_images - generated_images)
        pixel_loss = tf.reduce_mean(curr_proj_error * tf.expand_dims(curr_exp[:,:,:,1], -1))
        return pixel_loss


    def __init__(self, batch_size, trainable):
        self.batch_size = batch_size
        self.trainable = trainable

        self.input_imgs = tf.placeholder(tf.float32, shape = [None, 224, 224, 3], name = "input_imgs")
        self.real_imgs = tf.placeholder(tf.float32, shape = [None, 224, 224, 3], name = "real_imgs")
        
        # mean = [104, 117, 123]
        # scale_size = (224,224)
        
        # self.mean = tf.constant([104, 117, 123], dtype=tf.float32)
        # self.spec = [mean, scale_size]

        self.model()

        ##assign
        ##assert and cast them to same size!!!!
        self.generated=self.net_layers['predImg']
        print('.......')
        # print(self.generated.get_shape())
        with tf.name_scope("loss"):
          self.loss = self.reconstruction_loss(self.generated, self.real_imgs, self.net_layers['deconv_mask'])


        tf.summary.scalar('loss', self.loss)
