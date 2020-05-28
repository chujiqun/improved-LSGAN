import os, time, multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from glob import glob
from data import get_mnist, flags
from model import get_generator, get_discriminator

num_tiles = int(np.sqrt(flags.sample_size))

def GAN_loss(d_logits,d2_logits):
    """
    calculate the loss function of vanialla GAN
    :param d_logits: The output logits of fake data
    :param d2_logits: The output logits of true data
    :return: g_loss,d_loss
    """
    # discriminator: real images are labelled as 1
    d_loss_real = tl.cost.sigmoid_cross_entropy(d2_logits, tf.ones_like(d2_logits), name='dreal')
    # discriminator: images from generator (fake) are labelled as 0
    d_loss_fake = tl.cost.sigmoid_cross_entropy(d_logits, tf.zeros_like(d_logits), name='dfake')
    # combined loss for updating discriminator
    d_loss = d_loss_real + d_loss_fake
    # generator: reciprocal of d_loss_fake, which is vanilla GAN
    g_loss = -1*tl.cost.sigmoid_cross_entropy(d_logits, tf.zeros_like(d_logits), name='gfake')

    return g_loss,d_loss

def imporved_GAN_loss(d_logits,d2_logits):
    """
    calculate the objective function of improved-GAN
    :param d_logits: The output logits of fake data
    :param d2_logits: The output logits of true data
    :return: g_loss,d_loss
    """
    # discriminator: real images are labelled as 1
    d_loss_real = tl.cost.sigmoid_cross_entropy(d2_logits, tf.ones_like(d2_logits), name='dreal')
    # discriminator: images from generator (fake) are labelled as 0
    d_loss_fake = tl.cost.sigmoid_cross_entropy(d_logits, tf.zeros_like(d_logits), name='dfake')
    # combined loss for updating discriminator
    d_loss = d_loss_real + d_loss_fake
    # generator: try to fool discriminator to output 1, which is improved GAN
    g_loss = tl.cost.sigmoid_cross_entropy(d_logits, tf.ones_like(d_logits), name='gfake')

    return g_loss,d_loss

def WGAN_loss(d_logits,d2_logits):
    """
    calculate the objective function of WGAN
    :param d_logits: The output logits of fake data
    :param d2_logits: The output logits of true data
    :return: g_loss,d_loss
    """
    # discriminator: real images as much as possible
    d_loss_real = tf.reduce_mean(-1*d2_logits,name='dreal')
    # discriminator: images from generator (fake) as little as possible
    d_loss_fake = tf.reduce_mean(d_logits, name='dfake')
    # combined loss for updating discriminator
    d_loss = d_loss_real + d_loss_fake
    # generator: try to improve the scores of generated images.
    g_loss = tf.reduce_mean(-1*d_logits,name='gfake')

    return g_loss,d_loss

def WGAN_gp_loss(d_logits,d2_logits,gradient_penalty):
    """
    calculate the objective function of WGAN-gp
    :param d_logits: The output logits of fake data
    :param d2_logits: The output logits of true data
    :param gradient_penalty: L2 norm of the Discriminator Gradient
    :return: g_loss,d_loss
    """
    # discriminator: real images as much as possible
    d_loss_real = tf.reduce_mean(-1*d2_logits,name='dreal')
    # discriminator: images from generator (fake) as little as possible
    d_loss_fake = tf.reduce_mean(d_logits, name='dfake')
    # combined loss for updating discriminator
    d_loss = d_loss_real + d_loss_fake+gradient_penalty
    # generator: try to improve the scores of generated images.
    g_loss = tf.reduce_mean(-1*d_logits,name='gfake')

    return g_loss,d_loss

def LSGAN_loss(d_logits,d2_logits):
    """
    calculate the objective function of LSGAN, We set b=1, a=-1, c=0
    :param d_logits: The output logits of fake data
    :param d2_logits: The output logits of true data
    :return: g_loss,d_loss
    """
    # discriminator: real images as much as possible
    d_loss_real = tl.cost.mean_squared_error(d2_logits, tf.ones_like(d2_logits), name='dreal')
    # discriminator: images from generator (fake) as little as possible
    d_loss_fake = tl.cost.mean_squared_error(d_logits, -1*tf.ones_like(d_logits), name='dfake')
    # combined loss for updating discriminator
    d_loss = d_loss_real + d_loss_fake
    # generator: try to improve the scores of generated images.
    g_loss = tl.cost.mean_squared_error(d_logits, tf.zeros_like(d_logits), name='gfake')

    return g_loss,d_loss

def OurGAN_loss(d_logits,d2_logits):
    """
    calculate the objective function of LSGAN, We set b=1, a=-1, c=0, and lambda=1
    :param d_logits: The output logits of fake data
    :param d2_logits: The output logits of true data
    :return: g_loss,d_loss
    """
    # discriminator: real images as much as possible
    d_loss_real = tl.cost.mean_squared_error(d2_logits, tf.ones_like(d2_logits), name='dreal')
    # discriminator: images from generator (fake) as little as possible
    d_loss_fake = tl.cost.mean_squared_error(d_logits, -1*tf.ones_like(d_logits), name='dfake')
    # combined loss for updating discriminator
    d_loss = d_loss_real + d_loss_fake
    # generator: try to improve the scores of generated images.
    g_loss = tl.cost.mean_squared_error(d_logits, tf.zeros_like(d_logits), name='gfake')+1*tf.square(tf.reduce_mean(d2_logits)-tf.reduce_mean(d_logits))

    return g_loss,d_loss

methods_dict={"GAN": GAN_loss,
              "improved-GAN":imporved_GAN_loss,
              "WGAN":WGAN_loss,
              "LSGAN":LSGAN_loss,
              "WGAN-gp":WGAN_gp_loss,
              "our-GAN":OurGAN_loss}

class mutipletrainer(object):

    def __init__(self,flags,type):
        self.dataset,self.len_instance=get_mnist(flags.batch_size)
        self.G=get_generator([None, flags.z_dim], gf_dim=64, o_size=flags.output_size, o_channel=flags.c_dim)
        self.D=get_discriminator([None, flags.output_size, flags.output_size, flags.c_dim], df_dim=64)
        self.batch_size=flags.batch_size
        self.epoch=flags.n_epoch
        self.type=type
        assert type in methods_dict.keys()
        self.get_loss=methods_dict[type]
        if type=="WGAN":
            self.d_optimizer=tf.optimizers.RMSprop(flags.lr)
            self.g_optimizer=tf.optimizers.RMSprop(flags.lr)
        else:
            self.d_optimizer=tf.optimizers.Adam(flags.lr, beta_1=flags.beta1)
            self.g_optimizer=tf.optimizers.Adam(flags.lr, beta_1=flags.beta1)

    def train(self):
        self.G.train()
        self.D.train()
        n_step_epoch = int(self.len_instance // self.batch_size)
        for epoch in range(self.epoch):
            for step, batch_images in enumerate(self.dataset):
                if batch_images.shape[0] != self.batch_size:  # if the remaining data in this epoch < batch_size
                    break
                z = np.random.normal(loc=0.0, scale=1.0, size=[self.batch_size, flags.z_dim]).astype(np.float32)
                with tf.GradientTape(persistent=True) as tape:
                    d_logits = self.D(self.G(z))
                    d2_logits =self.D(batch_images)
                    if self.type=="WGAN-gp":
                        alpha=tf.random.uniform(shape=[self.batch_size,1,1,1],minval=0.,maxval=1.)
                        tmp_image=alpha*self.G(z)+(1-alpha)*batch_images
                        grads=tape.gradient(self.D(tmp_image,reuse=True),tmp_image)
                        gp=tf.reduce_mean((tf.sqrt(tf.reduce_sum(tf.square(tf.reshape(grads,[self.batch_size,-1])), 1))-1.)**2)
                        g_loss,d_loss=self.get_loss(d_logits,d2_logits,gp)
                    else:
                        g_loss,d_loss=self.get_loss(d_logits,d2_logits)

                grad = tape.gradient(g_loss, self.G.trainable_weights)
                self.g_optimizer.apply_gradients(zip(grad, self.G.trainable_weights))
                grad = tape.gradient(d_loss, self.D.trainable_weights)
                self.d_optimizer.apply_gradients(zip(grad, self.D.trainable_weights))
                if self.type == "WGAN":
                    for d in self.D.trainable_weights:
                        d.assign(tf.clip_by_value(d,-0.01,0.01))
                del tape
                print("Epoch: [{}/{}] [{}/{}], d_loss: {:.5f}, g_loss: {:.5f}".format(epoch,flags.n_epoch,step, n_step_epoch,d_loss, g_loss))

            if np.mod(epoch, flags.save_every_epoch) == 0:
                self.G.save_weights('{}/G.npz'.format(flags.checkpoint_dir), format='npz')
                self.D.save_weights('{}/D.npz'.format(flags.checkpoint_dir), format='npz')
                self.G.eval()
                result = self.G(z)
                self.G.train()
                tl.visualize.save_images(result.numpy(), [num_tiles, num_tiles],
                                         '{}/train_{}_{:02d}.png'.format(flags.sample_dir,self.type, epoch))

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # M=mutipletrainer(flags,type="GAN")
    M=mutipletrainer(flags,type="improved-GAN")
    # M=mutipletrainer(flags,type="WGAN")
    # M=mutipletrainer(flags,type="WGAN-gp")
    # M=mutipletrainer(flags,type="LSGAN")
    # M=mutipletrainer(flags,type="our-GAN")
    M.train()





if __name__ == '__main__':
    main()
