import numpy as np
import re
import itertools
from collections import Counter
import numpy as np
import time
import gc
import gzip
from random import random
import random
import sys
from scipy import misc
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
import os
#reload(sys)
#sys.setdefaultencoding("utf-8")
import numpy.linalg as linalg
from transformations import euler_from_matrix



class InputHelper(object):

    def setup(self,training_folder_path, objects):
        self.training_folder_path = training_folder_path
        self.objects = objects

    def get_single_view_info(self, batch_size):

        imgpaths_src=[]
        imgpaths_tgt=[]
        tforms_imgs=[]

        for x in range(batch_size):
            random_obj = np.random.randint(1,len(self.objects))
            start_angle = random.choice([5*i for i in range(72)])
            end_angle = random.choice([(start_angle+30*(i+1))%360 for i in range(12)])
            rotate_degree = end_angle - start_angle
            
            if (rotate_degree < 0):
                rotate_degree = 360 + rotate_degree
            if(rotate_degree == 0):
                rotate_degree = 360
            
            tf_info = [rotate_degree/360 for i in range(19)]
            tforms_imgs.append(tf_info)

            src_img_path=os.path.join(self.training_folder_path, str(random_obj), str(random_obj)+'_r'+str(start_angle)+'.png')
            tgt_img_path=os.path.join(self.training_folder_path, str(random_obj), str(random_obj)+'_r'+str(end_angle)+'.png')
            imgpaths_src.append(src_img_path)
            imgpaths_tgt.append(tgt_img_path)

        # print([imgpaths_src])
        # print(tforms_imgs)
        return imgpaths_src, tforms_imgs, imgpaths_tgt

    def get_multivw_info(self, batch_size, conv_model_spec ):

        imgpaths_src=[[] for i in range(2)]
        imgpaths_tgt=[]
        tforms_imgs=[[] for i in range(2)]

        seq_path = self.kitti_parentpath+"%02d/" % (seq_num,)
        odomlist=self.odomDict[seq_num]

        for x in range(batch_size):

            src_img_num = np.random.randint(0,seq_imgs_num)
            radius_num_a = np.random.randint(1,int(sample_range/2)+1)
            radius_num_b = np.random.randint(1,int(sample_range/2)+1)
            odom_src = odomlist[src_img_num]
            odom_src = np.reshape(odom_src,(3,4))

            if random()>0.5:
                if (src_img_num - sample_range) > 2:
                    tgt_img_num = src_img_num - radius_num_a
                    aux_img_num = tgt_img_num - radius_num_b
                else:
                    tgt_img_num = src_img_num + radius_num_a
                    aux_img_num = tgt_img_num + radius_num_b


            else:
                if (src_img_num + sample_range ) < seq_imgs_num - 3:
                    tgt_img_num = src_img_num + radius_num_a
                    aux_img_num = tgt_img_num + radius_num_b

                else:
                    tgt_img_num = src_img_num - radius_num_a
                    aux_img_num = tgt_img_num - radius_num_b


            odom_tgt=odomlist[tgt_img_num]
            odom_tgt=np.reshape(odom_tgt,(3,4))

            odom_aux=odomlist[aux_img_num]
            odom_aux=np.reshape(odom_aux,(3,4))

            newrow = [0,0,0,1]
            odom_tgt_4x4 = np.vstack([odom_tgt, newrow])
            odom_src_4x4 = np.vstack([odom_src, newrow])
            odom_aux_4x4 = np.vstack([odom_aux, newrow])

            src_matrices = []
            src_matrices.append(odom_src_4x4)
            src_matrices.append(odom_aux_4x4)

            for mat in src_matrices:

                odom_src_inv=linalg.inv(mat)
                #src inv * tgt = relative transform
                rel_odom_src_tgt=np.matmul(odom_src_inv,odom_tgt_4x4)
                ##converting to euler to get 6d =(3+3) dimensional vector for pose
                rel_tform_rot = rel_odom_src_tgt[0:3,0:3]
                rx,ry,rz = euler_from_matrix(rel_tform_rot)
                rel_tform_vec = [ rel_odom_src_tgt[0,3], rel_odom_src_tgt[1,3], rel_odom_src_tgt[2,3], rx, ry, rz]
                heightwise = np.tile(rel_tform_vec,(conv_model_spec[1][0]*conv_model_spec[1][1],1))
                widthwise = np.reshape(heightwise, (conv_model_spec[1][0],conv_model_spec[1][1],-1))
                tforms_imgs[src_matrices.index(mat)].append(widthwise)

            src_img_path=seq_path+ 'image_2/' +'%06d.png' % (src_img_num,)
            aux_img_path=seq_path+ 'image_2/' +'%06d.png' % (aux_img_num,)
            tgt_img_path=seq_path+ 'image_2/' +'%06d.png' % (tgt_img_num,)
            imgpaths_src[0].append(src_img_path)
            imgpaths_src[1].append(aux_img_path)
            imgpaths_tgt.append(tgt_img_path)

        

        return imgpaths_src, tforms_imgs, imgpaths_tgt




    def getInputBatch(self,batch_size, data, is_train, conv_model_spec, epoch,  get_img_tforms=1, is_multi_view=False):
        # random_obj = np.random.randint(0,len(data))
        # start_angle = random.choice([5*i for i in range(72)])
        # end_angle = random.choice([(start_angle+30*(i+1))%360 for i in range(12)])
        

        if(is_multi_view):
            imgpaths_src, tforms_imgs, imgpaths_tgt = self.get_multivw_info(batch_size, conv_model_spec)
        else:
            imgpaths_src, tforms_imgs, imgpaths_tgt = self.get_single_view_info( batch_size)

        src_imgslist = []

        src_imgslist.append(self.image_preprocess(imgpaths_src))

        tgt_imgslist = self.image_preprocess(imgpaths_tgt)

        return src_imgslist,tgt_imgslist,tforms_imgs



    def image_preprocess(self, img_paths):
        img_batch = []

        for img_path in img_paths:
            img_org = misc.imread(img_path)
            # print(np.asarray(img_org).shape)
            img_org = misc.imresize(img_org,[224,224,3])
            # img_normalized = self.normalize_input(img_org, conv_model_spec,crop_window)
            img_batch.append(img_org)

        #misc.imsave('temp1.png', np.vstack([np.hstack(batch1_seq),np.hstack(batch2_seq)]))

        temp =  np.asarray(img_batch)
        # temp = np.rollaxis(temp, 3, 1)
        # temp = temp.reshape(1,224, 224, 3)
        # temp = temp.reshape(1, 224,224,3)
        return temp


    # def batch_iter(self, x1, x2, y, video_lengths, batch_size, num_epochs, conv_model_spec, shuffle=True, is_train=True):
    #     """
    #     Generates a batch iterator for a dataset.
    #     """
    #     data_size = len(y)
    #     temp = int(data_size/batch_size)
    #     num_batches_per_epoch = temp+1 if (data_size%batch_size) else temp

    #     for epoch in range(num_epochs):
    #         # Shuffle the data at each epoch
    #         if shuffle:
    #             shuffle_indices = np.random.permutation(np.arange(data_size))
    #             x1_shuffled=x1[shuffle_indices]
    #             x2_shuffled=x2[shuffle_indices]
    #             y_shuffled=y[shuffle_indices]
    #             video_lengths_shuffled = video_lengths[shuffle_indices]
    #         else:
    #             x1_shuffled=x1
    #             x2_shuffled=x2
    #             y_shuffled=y
    #             video_lengths_shuffled = video_lengths
    #         for batch_num in range(num_batches_per_epoch):
    #             start_index = batch_num * batch_size
    #             end_index = min((batch_num + 1) * batch_size, data_size)
    #             #print(y_shuffled[start_index:end_index])

    #             processed_imgs = self.load_preprocess_images(x1_shuffled[start_index:end_index], x2_shuffled[start_index:end_index], conv_model_spec, epoch ,is_train)
    #             yield( processed_imgs[0], processed_imgs[1]  , y_shuffled[start_index:end_index], video_lengths_shuffled[start_index:end_index])


    def normalize_input(self, img, conv_model_spec,crop_window):
        img = img.astype(dtype=np.float32)
        #img = img[:, :, [2, 1, 0]] # swap channel from RGB to BGR
        vert_offset=100
        horz_offset=0
        img = img[vert_offset:vert_offset+conv_model_spec[1][0],crop_window*conv_model_spec[1][1]:crop_window*conv_model_spec[1][1]+conv_model_spec[1][1]]
        img = img - conv_model_spec[0]

        return img

    # Data Preparatopn
    # ==================================================


    """
    def apply_image_augmentations(self):
        sometimes = lambda aug: iaa.Sometimes(0.33, aug)
        self.train_seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            sometimes(iaa.Crop(percent=(0, 0.05))), # crop images by 0-5% of their height/width
            sometimes(iaa.Affine(
                scale={"x": (0.9, 1.21), "y": (0.9, 1.1)}, # scale images to 90-110% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                shear=(-10, 10), # shear by -12 to +12 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((2, 5),
                [
                    #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 1.5)), # emboss images
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    iaa.Invert(0.05, per_channel=True), # invert color channels
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.Multiply((0.5, 1.5), per_channel=0.5), # change brightness of images (50-150% of original value)
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 4.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.03))) # sometimes move parts of the image around
                ],
                random_order=True
            )
        ],
        random_order=True
        )

    def data_augmentations(self):
        seq_det = []
        for i in range(5):
            seq_det.append(self.train_seq.to_deterministic())
        self.seq_det = seq_det


    """

def save_plot(val1, val2, xlabel, ylabel, title, axis, legend,path):
    pyplot.figure()
    pyplot.plot(val1, '*r--', val2, '^b-')
    pyplot.xlabel(xlabel)
    pyplot.ylabel(ylabel)
    pyplot.title(title)
    pyplot.axis(axis)
    pyplot.legend(legend)
    pyplot.savefig(path+'.pdf')
    pyplot.clf()

def compute_distance(distance, loss):
    d = np.copy(distance)
    if loss == "AAAI":
        d[distance>=0.5]=1
        d[distance<0.5]=0
    elif loss == "contrastive":
        d[distance>0.5]=0
        d[distance<=0.5]=1
    else:
        raise ValueError("Unkown loss function {%s}".format(loss))
    return d
