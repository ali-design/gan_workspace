from __future__ import division
from __future__ import print_function
import os
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

from ops import *
from utils import *

import io
import IPython.display
import PIL.Image

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

def gen_random(mode, size):
    if mode=='normal01': return np.random.normal(0,1,size=size)
    if mode=='uniform_signed': return np.random.uniform(-1,1,size=size)
    if mode=='uniform_unsigned': return np.random.uniform(0,1,size=size)


class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         max_to_keep=1,
         input_fname_pattern='*.jpg', checkpoint_dir='ckpts', sample_dir='samples', out_dir='./out', data_dir='./data'):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """    
    self.sess = sess
    self.crop = crop

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    if not self.y_dim:
      self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    if not self.y_dim:
      self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.data_dir = data_dir
    self.out_dir = out_dir
    self.max_to_keep = max_to_keep

    if self.dataset_name == 'mnist':
      self.data_X, self.data_y = self.load_mnist()
      self.c_dim = self.data_X[0].shape[-1]
    else:
      data_path = os.path.join(self.data_dir, self.dataset_name, self.input_fname_pattern)
      self.data = glob(data_path)
      if len(self.data) == 0:
        raise Exception("[!] No data found in '" + data_path + "'")
      np.random.shuffle(self.data)
      imreadImg = imread(self.data[0])
      if len(imreadImg.shape) >= 3: #check if image is a non-grayscale image by checking channel number
        self.c_dim = imread(self.data[0]).shape[-1]
      else:
        self.c_dim = 1

      if len(self.data) < self.batch_size:
        raise Exception("[!] Entire dataset size is less than the configured batch_size")
    
    self.grayscale = (self.c_dim == 1)

    self.build_model()

  def build_model(self):
    # Lets first define our placeholders and vars:
    self.img_size = 28
    self.Nsliders = 1
    img_size = self.img_size
    Nsliders = self.Nsliders
    
    ## Placeholder for new z
    self.target = tf.placeholder(tf.float32, shape=(None, img_size, img_size, Nsliders))
    self.mask = tf.placeholder(tf.float32, shape=(None, img_size, img_size, Nsliders))
    self.alpha = tf.placeholder(tf.float32, shape=(None, self.Nsliders))
    self.w = tf.Variable(np.random.uniform(-1, 1, [1, self.z_dim]), name='walk', dtype=np.float32)
    ######
    
    # The rest is the original code:
    if self.y_dim:
#       self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
      self.y = tf.placeholder(tf.float32, shape=(None, self.y_dim), name='y')

    else:
      self.y = None

    if self.crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)
    
    self.G                  = self.generator(self.z, self.y, reuse=False)
    self.D, self.D_logits   = self.discriminator(inputs, self.y, reuse=False)
    self.sampler            = self.sampler(self.z, self.y)
    self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)
    
    ## This is new z    
    self.z_new = self.z + self.alpha * self.w
#     self.z_new_sum = histogram_summary("z_new", self.z_new)
    
    self.G_new                      = self.generator(self.z_new, self.y, reuse=True)
    self.sampler_new                = self.my_sampler(self.z_new, self.y)
    self.D_new_, self.D_logits_new_ = self.discriminator(self.G_new, self.y, reuse=True)
    ######
    
    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.G_sum = image_summary("G", self.G)
    
    ## This is for new z
#     self.d_new_sum = histogram_summary("d_new", self.D_new_)
#     self.G_new_sum = histogram_summary("G_new", self.G_new)

    def sigmoid_cross_entropy_with_logits(x, y):
      try:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
      except:
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.g_loss = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
    
    ## This is for new z
    self.d_loss_fake_new = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits_new_, tf.zeros_like(self.D_new_)))
    self.g_loss_new =tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(self.D_logits_new_, tf.ones_like(self.D_new_)))
    self.walk_loss = self.g_loss + self.g_loss_new + tf.losses.compute_weighted_loss(tf.square(self.G_new - self.target), weights=self.mask)
    ######

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
                          
#     self.d_loss = self.d_loss_real + self.d_loss_fake
    ## This is for new z
    self.d_loss = self.d_loss_real + self.d_loss_fake + self.d_loss_fake_new
#     self.g_loss = self.g_loss + self.g_loss_new
    self.g_loss = self.walk_loss
    ######

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]
    
    self.saver = tf.train.Saver(max_to_keep=self.max_to_keep)

  def imgrid(self, imarray, cols=5, pad=1):
    if imarray.dtype != np.uint8:
      raise ValueError('imgrid input imarray must be uint8')
    pad = int(pad)
    assert pad >= 0
    cols = int(cols)
    assert cols >= 1
    N, H, W, C = imarray.shape
    rows = int(np.ceil(N / float(cols)))
    batch_pad = rows * cols - N
    assert batch_pad >= 0
    post_pad = [batch_pad, pad, pad, 0]
    pad_arg = [[0, p] for p in post_pad]
    imarray = np.pad(imarray, pad_arg, 'constant', constant_values=255)
    H += pad
    W += pad
    grid = (imarray
            .reshape(rows, cols, H, W, C)
            .transpose(0, 2, 1, 3, 4)
            .reshape(rows*H, cols*H, C))
    if pad:
      grid = grid[:-pad, :-pad]
    return grid

  def imshow(self, a, im_size=128, format='png', jpeg_fallback=True, filename=None):
    if a.dtype != np.uint8:
      a = a*255
    a = np.asarray(a, dtype=np.uint8)
    a = cv2.resize(a, (a.shape[1], a.shape[0]))

    str_file = io.BytesIO()
    PIL.Image.fromarray(a).save(str_file, format)
    im_data = str_file.getvalue()
    try:
      disp = IPython.display.display(IPython.display.Image(im_data))
      if filename:
          size = (a.shape[1]//2, a.shape[0]//2)
          im = PIL.Image.fromarray(a)
          im.thumbnail(size,PIL.Image.ANTIALIAS)
          im.save('{}.{}'.format(filename, format))

    except IOError:
      if jpeg_fallback and format != 'jpeg':
        print ('Warning: image was too large to display in format "{}"; '
               'trying jpeg instead.').format(format)
        return imshow(a, format='jpeg')
      else:
        raise
    return disp

  def get_target_np(self, outputs_zs, alpha, show_img=False, show_mask=False):
    target_fn = np.copy(outputs_zs)
    mask_fn = np.ones(outputs_zs.shape)
    mask_out = np.ones(outputs_zs.shape)

#     img_size = self.img_size
    img_size = 28
    
    for b in range(outputs_zs.shape[0]):
        print('alpha: ', alpha[b,0])
        if alpha[b,0] !=1:
            new_size = int(alpha[b,0]*img_size)
    
            ## crop            
            if alpha[b,0] < 1:
                print('alpha < 1 => crop')
                output_cropped[b,:,:,:] = outputs_zs[b,img_size//2-new_size//2:img_size//2+new_size//2, img_size//2-new_size//2:img_size//2+new_size//2,:]
                mask_cropped[b,:,:,:] = mask_fn[b,:,:,:]
            ## padding
            else:
                print('alpha > 1 => pad')
                output_cropped = np.zeros((1, new_size, new_size, outputs_zs.shape[3]))
                mask_cropped = np.zeros((1, new_size, new_size, outputs_zs.shape[3]))
                output_cropped[b, new_size//2-img_size//2:new_size//2+img_size//2, new_size//2-img_size//2:new_size//2+img_size//2,:] = outputs_zs 
                mask_cropped[b, new_size//2-img_size//2:new_size//2+img_size//2, new_size//2-img_size//2:new_size//2+img_size//2,:] = mask_fn

            ## Resize
            target_fn[b,:,:,:] = np.zeros(1, outputs_zs.shape[1], outputs_zs.shape[2], outputs_zs.shape[3])
            mask_out[b,:,:,:] = np.zeros(1, outputs_zs.shape[1], outputs_zs.shape[2], outputs_zs.shape[3])
            for i in range(outputs_zs.shape[0]):
                target_fn[i,:,:,:] = np.expand_dims(cv2.resize(output_cropped[i,:,:,:], (img_size, img_size), interpolation = cv2.INTER_LINEAR), axis=2)
                mask_out[i,:,:,:] = np.expand_dims(cv2.resize(mask_cropped[i,:,:,:], (img_size, img_size), interpolation = cv2.INTER_LINEAR), axis=2)

            mask_out[np.nonzero(mask_out)] = 1.
            assert(np.setdiff1d(mask_out, [0., 1.]).size == 0)
                
#             M = np.float32([[1,0,alpha[i,0]],[0,1,0]])
#             target_fn[i,:,:,:] = np.expand_dims(cv2.warpAffine(outputs_zs[i,:,:,:], M, (self.img_size, self.img_size)), axis=2)
#             mask_fn[i,:,:,:] = np.expand_dims(cv2.warpAffine(mask_fn[i,:,:,:], M, (self.img_size, self.img_size)), axis=2)

#     mask_fn[np.nonzero(mask_fn)] = 1.
#     assert(np.setdiff1d(mask_fn, [0., 1.]).size == 0)
        
#     print('target_fn.shape', target_fn.shape)
    if show_img:
        print('Target image:')
        self.imshow(self.imgrid(np.uint8(target_fn), cols=outputs_zs.shape[0]))

    if show_mask:
        print('Target mask:')
        self.imshow(self.imgrid(np.uint8(mask_out), cols=outputs_zs.shape[0]))

    return target_fn, mask_out

  def train(self, config):
    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    # define optimal walk loss:
    # in gan_steerability w_optim is "train_step", and we move definintion of loss to build_model fn, 
    # so instead of "loss", we call it walk_loss
#     loss = tf.losses.compute_weighted_loss(tf.square(transformed_output-target), weights=mask)
    lr = config.learning_rate * 1
    w_optim = tf.train.AdamOptimizer(lr).minimize(self.walk_loss, var_list=tf.trainable_variables(scope='walk'), 
                                                 name='AdamOpter')
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    if config.G_img_sum:
      self.g_sum = merge_summary([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
    else:
      self.g_sum = merge_summary([self.z_sum, self.d__sum, self.d_loss_fake_sum, self.g_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
    self.writer = SummaryWriter(os.path.join(self.out_dir, "logs"), self.sess.graph)

    sample_z = gen_random(config.z_dist, size=(self.sample_num , self.z_dim))
    
    if config.dataset == 'mnist':
      sample_inputs = self.data_X[0:self.sample_num]
      sample_labels = self.data_y[0:self.sample_num]
    else:
      sample_files = self.data[0:self.sample_num]
      sample = [
          get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    crop=self.crop,
                    grayscale=self.grayscale) for sample_file in sample_files]
      if (self.grayscale):
        sample_inputs = np.array(sample).astype(np.float32)[:, :, :, None]
      else:
        sample_inputs = np.array(sample).astype(np.float32)
  
    counter = 1
    start_time = time.time()
    could_load, checkpoint_counter = self.load(self.checkpoint_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
      if config.dataset == 'mnist':
        batch_idxs = min(len(self.data_X), config.train_size) // config.batch_size
      else:      
        self.data = glob(os.path.join(
          config.data_dir, config.dataset, self.input_fname_pattern))
        np.random.shuffle(self.data)
        batch_idxs = min(len(self.data), config.train_size) // config.batch_size

      for idx in xrange(0, int(batch_idxs)):
        if config.dataset == 'mnist':
          batch_images = self.data_X[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_labels = self.data_y[idx*config.batch_size:(idx+1)*config.batch_size]
        else:
          batch_files = self.data[idx*config.batch_size:(idx+1)*config.batch_size]
          batch = [
              get_image(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        crop=self.crop,
                        grayscale=self.grayscale) for batch_file in batch_files]
          if self.grayscale:
            batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
          else:
            batch_images = np.array(batch).astype(np.float32)

        batch_z = gen_random(config.z_dist, size=[config.batch_size, self.z_dim]) \
              .astype(np.float32)

        if config.dataset == 'mnist':
            
#           alpha_vals = np.random.randint(-5, 6, size=[config.batch_size,1])  
# #           alpha_vals = np.zeros([config.batch_size,1])
# #           test_alpha, test_w = self.sess.run([self.alpha, self.w], feed_dict={self.alpha: alpha_vals})
#           out_zs = self.sampler.eval({ self.z: batch_z, self.y: batch_labels })

#           target_fn, mask_fn = self.get_target_np(out_zs, alpha_vals)#, show_img=True, show_mask=True)
          
# #           G_np = self.G_new.eval({self.z: batch_z, self.y:batch_labels, self.alpha: alpha_vals})
# #           G_new_np = self.G_new.eval({self.z: batch_z, self.y:batch_labels, self.alpha:alpha_vals})
# #           print('sum of g and g_new diff', np.sum(G_new_np - G_np)**2)
# #           print(np.all(G_new_np==G_np))
            
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={ 
              self.inputs: batch_images,
              self.z: batch_z,
              self.y:batch_labels,
              self.target:target_fn,
              self.mask:mask_fn,
              self.alpha:alpha_vals
            })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          summary_w_optim = self.sess.run(w_optim,
            feed_dict={
              self.z: batch_z, 
              self.y:batch_labels,
              self.target:target_fn,
              self.mask:mask_fn,
              self.alpha:alpha_vals                
            })
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={
              self.z: batch_z, 
              self.y:batch_labels,
              self.target:target_fn,
              self.mask:mask_fn,
              self.alpha:alpha_vals                
            })
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          summary_w_optim = self.sess.run(w_optim,
            feed_dict={ self.z: batch_z, self.y:batch_labels, self.target:target_fn,
                        self.mask:mask_fn,
                        self.alpha:alpha_vals})              
          _, summary_str, summary_w_optim = self.sess.run([g_optim, self.g_sum, w_optim],
            feed_dict={ self.z: batch_z, self.y:batch_labels, self.target:target_fn,
                        self.mask:mask_fn,
                        self.alpha:alpha_vals})
          self.writer.add_summary(summary_str, counter)
          
          errD_fake = self.d_loss_fake.eval({
              self.z: batch_z, 
              self.y:batch_labels,
              self.target:target_fn,
              self.mask:mask_fn,
              self.alpha:alpha_vals              
          })
          errD_real = self.d_loss_real.eval({
              self.inputs: batch_images,
              self.y:batch_labels,
              self.target:target_fn,
              self.mask:mask_fn,
              self.alpha:alpha_vals              
          })
          errWalk = self.walk_loss.eval({
              self.z: batch_z,
              self.y: batch_labels,
              self.target:target_fn,
              self.mask:mask_fn,
              self.alpha:alpha_vals
          })
          errG = self.g_loss.eval({
              self.z: batch_z,
              self.y: batch_labels,
              self.target:target_fn,
              self.mask:mask_fn,
              self.alpha:alpha_vals
          })
        else:
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={ self.inputs: batch_images, self.z: batch_z })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str, summary_w_optim = self.sess.run([g_optim, self.g_sum, w_optim],
            feed_dict={ self.z: batch_z })
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          _, summary_str, summary_w_optim = self.sess.run([g_optim, self.g_sum, w_optim],
            feed_dict={ self.z: batch_z })
          self.writer.add_summary(summary_str, counter)
          
          errD_fake = self.d_loss_fake.eval({ self.z: batch_z })
          errD_real = self.d_loss_real.eval({ self.inputs: batch_images })
          errG = self.g_loss.eval({self.z: batch_z})

        print("[%8d Epoch:[%2d/%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, walk_loss: %.8f" \
          % (counter, epoch, config.epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG, errWalk))

        if np.mod(counter, config.sample_freq) == 0:
          if config.dataset == 'mnist':
            sample_alpha = np.random.randint(-5, 6, size=[config.batch_size,1])  
#             sample_alpha = np.zeros([config.batch_size,1])
            sample_out_zs = self.sampler.eval({ self.z: sample_z, self.y: sample_labels })
            sample_target_fn, sample_mask_fn = self.get_target_np(sample_out_zs, sample_alpha)#, show_img=True, show_mask=True)
            
            samples, d_loss, g_loss, walk_loss = self.sess.run(
              [self.sampler, self.d_loss, self.g_loss, self.walk_loss],
              feed_dict={
                  self.z: sample_z,
                  self.inputs: sample_inputs,
                  self.y:sample_labels,
                  self.target: sample_target_fn,
                  self.mask: sample_mask_fn,
                  self.alpha: sample_alpha
              }
            )
            save_images(samples, image_manifold_size(samples.shape[0]),
                  './{}/train_{:08d}.png'.format(config.sample_dir, counter))
#             print("[Sample] d_loss: %.8f, g_loss: %.8f, w_loss: %.8f" % (d_loss, g_loss, walk_loss)) 
            print("[Sample] d_loss: {}, g_loss: {}, w_loss: {}".format(d_loss, g_loss, walk_loss))
          else:
            try:
              samples, d_loss, g_loss = self.sess.run(
                [self.sampler, self.d_loss, self.g_loss],
                feed_dict={
                    self.z: sample_z,
                    self.inputs: sample_inputs,
                },
              )
              save_images(samples, image_manifold_size(samples.shape[0]),
                    './{}/train_{:08d}.png'.format(config.sample_dir, counter))
              print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss)) 
            except:
              print("one pic error!...")

        if np.mod(counter, config.ckpt_freq) == 0:
          self.save(config.checkpoint_dir, counter)
        
        counter += 1
        
  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      if not self.y_dim:
        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h4_lin')

        return tf.nn.sigmoid(h4), h4
      else:
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        x = conv_cond_concat(image, yb)

        h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
        h0 = conv_cond_concat(h0, yb)

        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))
        h1 = tf.reshape(h1, [self.batch_size, -1])      
        h1 = concat([h1, y], 1)
        
        h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin')))
        h2 = concat([h2, y], 1)

        h3 = linear(h2, 1, 'd_h3_lin')
        
        return tf.nn.sigmoid(h3), h3

  def generator(self, z, y=None, reuse=False):
    with tf.variable_scope("generator") as scope:
      if reuse:
        scope.reuse_variables()
        
      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        ## z and z_new here:
        print('In G, z.shape: ', z.shape)
        z_new = z + alpha * self.w
        print('In G, z_new.shape', z_new.shape)
        
        
        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(
            z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(
            self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = deconv2d(
            h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(
            h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(
            h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(
            h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)
      else:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)
        
        h0 = tf.nn.relu(
            self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
        h0 = concat([h0, y], 1)

        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin')))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])

        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
            [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2')))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(
            deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  def sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        h0 = tf.reshape(
            linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
            [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

        return tf.nn.tanh(h4)
      else:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)

        h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
        h0 = concat([h0, y], 1)

        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(self.g_bn2(
            deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  def my_sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        h0 = tf.reshape(
            linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
            [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

        return tf.nn.tanh(h4)
      else:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
        yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)

        h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=False))
        h0 = concat([h0, y], 1)

        h1 = tf.nn.relu(self.g_bn1(
            linear(h0, self.gf_dim*2*s_h4*s_w4, 'g_h1_lin'), train=False))
        h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
        h1 = conv_cond_concat(h1, yb)

        h2 = tf.nn.relu(self.g_bn2(
            deconv2d(h1, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2'), train=False))
        h2 = conv_cond_concat(h2, yb)

        return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

  def load_mnist(self):
    print('loading mnist...')
    data_dir = os.path.join(self.data_dir, self.dataset_name)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    # here shift train
    idx = np.random.choice(60000, 50000, replace=False)
    for i in range(len(idx)):
        img_test_5px = np.zeros([28,28], dtype= 'float')
        img_test = trX[idx[i],:,:,0]
        offset_val = np.random.randint(1, 6)  
        coin = np.random.uniform(0, 1)
        if coin <= 0.5:
            offset_val = -offset_val            
        if(offset_val > 0):
            img_test_5px[:,offset_val:] = img_test[:,:-offset_val]
        else:
            img_test_5px[:,:28+offset_val] = img_test[:,-offset_val:]
        trX[idx[i],:,:,0] = img_test_5px
    
    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)
    
    # here shfit test
    idx = np.random.choice(10000, 9000, replace=False)
    for i in range(len(idx)):
        img_test_5px = np.zeros([28,28], dtype= 'float')
        img_test = teX[idx[i],:,:,0]
        offset_val = np.random.randint(0, 6)  
        coin = np.random.uniform(0, 1)
        if coin <= 0.5:
            offset_val = -offset_val            
        if(offset_val > 0):
            img_test_5px[:,offset_val:] = img_test[:,:-offset_val]
        else:
            img_test_5px[:,:28+offset_val] = img_test[:,-offset_val:]
        teX[idx[i],:,:,0] = img_test_5px
    
    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)
    
    y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0
    
    return X/255.,y_vec

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)

  def save(self, checkpoint_dir, step, filename='model', ckpt=True, frozen=False):
    # model_name = "DCGAN.model"
    # checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    filename += '.b' + str(self.batch_size)
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    if ckpt:
      self.saver.save(self.sess,
              os.path.join(checkpoint_dir, filename),
              global_step=step)

    if frozen:
      tf.train.write_graph(
              tf.graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, ["generator_1/Tanh"]),
              checkpoint_dir,
              '{}-{:06d}_frz.pb'.format(filename, step),
              as_text=False)

  def load(self, checkpoint_dir):
    #import re
    print(" [*] Reading checkpoints...", checkpoint_dir)
    # checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
    # print("     ->", checkpoint_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      #counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      counter = int(ckpt_name.split('-')[-1])
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
