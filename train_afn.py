#! /usr/bin/env python
import sys
import tensorflow as tf
import numpy as np
import re
import os
import time
import datetime
import gc
from helper_aloi import InputHelper, save_plot
import gzip
from random import random
from model_single_view import Net
from model_multi_view import Net_MultiView
from scipy.misc import imsave
# Parameters
# ==================================================
tf.flags.DEFINE_string("training_folder_path", "masked_256/", "training folder")
tf.flags.DEFINE_string("name", "result", "prefix names of the output files(default: result)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 10)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many epochs (default: 100)")
tf.flags.DEFINE_string("loss", "contrastive", "Type of Loss function")
tf.flags.DEFINE_boolean("is_train", False, "Training ConvNet (Default: False)")
tf.flags.DEFINE_float("lr", 0.0001, "learning-rate(default: 0.00001)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", False, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("summaries_dir", "outputs/summaries/", "Summary storage")

#Model Parameters
tf.flags.DEFINE_string("checkpoint_path", "", "pre-trained checkpoint path")
tf.flags.DEFINE_integer("numObjects", 900, "number of objects")
tf.flags.DEFINE_integer("batches_train", 5400 , "batches for train")
tf.flags.DEFINE_integer("batches_seen_test", 675, "batches for seen test")
tf.flags.DEFINE_integer("batches_unseen_test", 675, "batches for unseen test")

tf.flags.DEFINE_boolean("conv_net_training", True, "Training ConvNet (Default: False)")
tf.flags.DEFINE_boolean("multi_view_training", False, "Training ConvNet (Default: False)")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.flag_values_dict().items()):
    print("{}={}".format(attr.upper(), value))
print("")

objects=[ i for i in range(1,FLAGS.numObjects+1) ] ##number stands for object inside ith folder
#break into train and test
objtrain=objects[0:800]#training from 1-800
obj_seen_test=objects[699:800]#seen test from 700-800
obj_unseen_test=objects[799:900]#unseen test from 800-900

inpH = InputHelper()
inpH.setup(FLAGS.training_folder_path,objects)


# Training
# ==================================================
print("starting graph def")
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement,
      gpu_options=gpu_options,
      )
    sess = tf.Session(config=session_conf)
    print("started session")
    with sess.as_default():
        if(FLAGS.multi_view_training):
            convModel = Net_MultiView(
                 FLAGS.batch_size,
                 FLAGS.conv_net_training)
        else:

            convModel = Net(
             FLAGS.batch_size,
             FLAGS.conv_net_training)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        #learning_rate=tf.train.exponential_decay(1e-5, global_step, sum_no_of_batches*5, 0.95, staircase=False, name=None)
        optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        print("initialized Net object")

    grads_and_vars=optimizer.compute_gradients(convModel.loss)
    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print("defined training_ops")
    # keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    summaries_merged = tf.summary.merge_all()
    print("defined gradient summaries")
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join("outputs/", "runs", FLAGS.name))
    print("Writing to {}\n".format(out_dir))

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    #current_path = os.getcwd()
    #imgdir_path=os.path.join(current_path,'imgs')
    #if not os.path.exists(imgdir_path):
        #os.makedirs(imgdir_path)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    #Fix weights for Conv Layers
    #convModel.initalize(sess)

    #print all trainable parameters
    tvar = tf.trainable_variables()
    for i, var in enumerate(tvar):
        print("{}".format(var.name))


    print("init all variables")
    graph_def = tf.get_default_graph().as_graph_def()
    graphpb_txt = str(graph_def)
    with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
        f.write(graphpb_txt)


    train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', graph=tf.get_default_graph())
    seen_test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/val' , graph=tf.get_default_graph())
    unseen_test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/val' , graph=tf.get_default_graph())

    def train_step(batch_size, src_batch, real_img_batch, tform_batch, train_iter, multi_view_training):

        #A single training step
        if(FLAGS.multi_view_training):

            feed_dict={convModel.input_imgs: src_batch[0],
                        convModel.aux_imgs: src_batch[1],
                        convModel.tgt_imgs: real_img_batch,
                        convModel.tform: tform_batch[0],
                        convModel.tform_aux: tform_batch[1] }
        else:

            feed_dict={convModel.input_imgs: src_batch[0],
                        convModel.real_imgs: real_img_batch[0],
                        convModel.tform: tform_batch}


        if(train_iter%FLAGS.batches_train==0):
            outputs, _, step, loss, summary = sess.run([convModel.generated, tr_op_set, global_step, convModel.loss, summaries_merged],  feed_dict)
            img_num=0
            for i in range(len(outputs)):
                imsave('outputs/imgs/'+str(train_iter)+'_'+str(img_num)+'_output.png', outputs[i])
                imsave('outputs/imgs/'+str(train_iter)+'_'+str(img_num)+'_target.png', real_img_batch[i])
                for j in range(len(src_batch)):
                    imsave('outputs/imgs/'+str(train_iter)+'_'+str(img_num)+'_input'+str(j)+'.png', src_batch[j][i])
                img_num+=1
        else:
             _, step, loss, summary = sess.run([tr_op_set, global_step, convModel.loss, summaries_merged],  feed_dict)
        time_str = datetime.datetime.now().isoformat()
        return summary, loss

    def dev_step(src_batch, real_img_batch, tform_batch, dev_iter, epoch, multi_view_training):

        #A single validation step

        if(FLAGS.multi_view_training):

            feed_dict={convModel.input_imgs: src_batch[0],
                        convModel.aux_imgs: src_batch[1],
                        convModel.tgt_imgs: real_img_batch,
                        convModel.tform: tform_batch[0],
                        convModel.tform_aux: tform_batch[1] }

        else:

            feed_dict={convModel.input_imgs: src_batch[0],
                        convModel.tgt_imgs: real_img_batch,
                        convModel.tform: tform_batch[0] }



        step, loss, summary, outputs= sess.run([global_step, convModel.loss, summaries_merged,convModel.tgts],  feed_dict)

        time_str = datetime.datetime.now().isoformat()

        return summary, loss

    if not os.path.exists('outputs/imgs/'):
        os.makedirs('outputs/imgs/')

    start_time = time.time()
    train_loss, val_loss = [], []
    train_batch_loss_arr, val_batch_loss_arr = [], []

    for nn in range(FLAGS.num_epochs):

        current_step = tf.train.global_step(sess, global_step)
        print("Epoch Number: {}".format(nn))
        epoch_start_time = time.time()
        train_epoch_loss=0.0
        for kk in range(FLAGS.batches_train):
            print(str(kk))
            src_batch, real_img_batch, tform_batch = inpH.getInputBatch(FLAGS.batch_size,objtrain,True, convModel.spec,nn, FLAGS.multi_view_training)
            summary, train_batch_loss =train_step(FLAGS.batch_size, src_batch, real_img_batch, tform_batch, kk, FLAGS.multi_view_training)
            train_writer.add_summary(summary, current_step)
            train_epoch_loss = train_epoch_loss + train_batch_loss* len(tform_batch)
            train_batch_loss_arr.append(train_batch_loss*len(tform_batch))
        print("train_loss ={}".format(train_epoch_loss/(FLAGS.batches_train*FLAGS.batch_size)))
        train_loss.append(train_epoch_loss/(FLAGS.batches_train*FLAGS.batch_size))

        # Evaluate on Validataion Data for every epoch
        # val_epoch_loss=0.0
        # print("\nEvaluation:")

        # for kk in range(FLAGS.batches_test):
        #     src_dev_b, tgt_dev_b, tform_dev_b = inpH.getInputBatch(FLAGS.batch_size,objstest,True, convModel.spec, nn, FLAGS.multi_view_training)

        #     summary,  val_batch_loss = dev_step(src_dev_b, tgt_dev_b, tform_dev_b, kk ,nn, FLAGS.multi_view_training)

        #     val_writer.add_summary(summary, current_step)
        #     val_epoch_loss = val_epoch_loss + val_batch_loss*len(tform_dev_b)
        #     val_batch_loss_arr.append(val_batch_loss*len(tform_dev_b))
        #     print("val_loss ={}".format(val_epoch_loss/FLAGS.batch_size*FLAGS.batches_test))
        # val_loss.append(val_epoch_loss/FLAGS.batch_size*FLAGS.batches_test)



        # Update stored model
        # if current_step % (FLAGS.checkpoint_every) == 0:
        #     saver.save(sess, checkpoint_prefix, global_step=current_step)
        #     tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph"+str(nn)+".pb", as_text=False)
        #     print("Saved model {} with checkpoint to {}".format(nn, checkpoint_prefix))

        # epoch_end_time = time.time()
        # empty=[]
        # print("Total time for {} th-epoch is {}\n".format(nn, epoch_end_time-epoch_start_time))
        # save_plot(train_loss, val_loss, 'epochs', 'loss', 'Loss vs epochs', [-0.1, nn+0.1, 0, np.max(train_loss)+0.2],  ['train','val' ],'./loss_'+str(FLAGS.name))
        # save_plot(train_batch_loss_arr, val_batch_loss, 'steps', 'loss', 'Loss vs steps', [-0.1, (nn+1)*FLAGS.batch_size*FLAGS.batches_test+0.1, 0, np.max(train_batch_loss_arr)+0.2],  ['train','val' ],'./loss_batch_'+str(FLAGS.name))

    end_time = time.time()
    print("Total time for {} epochs is {}".format(FLAGS.num_epochs, end_time-start_time))
