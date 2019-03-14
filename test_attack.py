## test_attack.py -- sample code to test attack procedure
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import time

from setup_cifar import CIFAR, CIFARModel
from setup_mnist import MNIST, MNISTModel
from setup_inception import ImageNet, InceptionModel

from l2_attack import CarliniL2
from l0_attack import CarliniL0
from li_attack import CarliniLi

import matplotlib.pyplot as plt
import random

def show(img):
    """
    Show MNSIT digits in the console.
    """
    remap = "  .*#"+"#"*100
    img = (img.flatten()+.5)*3
    if len(img) != 784: return
    print("START")
    for i in range(28):
        print("".join([remap[int(round(x))] for x in img[i*28:i*28+28]]))


def generate_data(data, samples, targeted=True, start=0, inception=False):
    """
    Generate the input data to the attack algorithm.

    data: the images to attack
    samples: number of samples to use
    targeted: if true, construct targeted attacks, otherwise untargeted attacks
    start: offset into data to use
    inception: if targeted and inception, randomly sample 100 targets intead of 1000
    """
    inputs = []
    targets = []
    for i in range(samples):
        if targeted:
            if inception:
                seq = random.sample(range(1,1001), 10)
            else:
                seq = range(data.test_labels.shape[1])

            for j in seq:
                if (j == np.argmax(data.test_labels[start+i])) and (inception == False):
                    continue
                inputs.append(data.test_data[start+i])
                targets.append(np.eye(data.test_labels.shape[1])[j])
        else:
            inputs.append(data.test_data[start+i])
            targets.append(data.test_labels[start+i])

    inputs = np.array(inputs)
    targets = np.array(targets)

    return inputs, targets


def generate_data_2(data):
    inputs = []
    targets = []
    nlabels = data.test_labels.shape[1]
    for i in range(nlabels):
        for j in range(1000):
            x = data.test_data[j]
            y = data.test_labels[j]
            if i == np.argmax(y):
                inputs.append(x)
                onehot = np.zeros(nlabels)
                onehot[(i+1) % 10] = 1
                targets.append(onehot)
                break
    inputs = np.array(inputs)
    targets = np.array(targets)
    return inputs, targets

def __test():
    generate_data_2(data)

if __name__ == "__main__":
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    with tf.Session() as sess:
        data, model =  MNIST(), MNISTModel("models/mnist", sess)
        #data, model =  CIFAR(), CIFARModel("models/cifar", sess)
        # attack = CarliniL2(sess, model, batch_size=9,
        #                    max_iterations=1000, confidence=0)
        attack = CarliniL0(sess, model, max_iterations=1000,
                           initial_const=10, largest_const=15)
        # attack = CarliniLi(sess, model)

        inputs, targets = generate_data(data, samples=1, targeted=True,
                                        start=0, inception=False)
        timestart = time.time()
        adv = attack.attack(inputs, targets)
        timeend = time.time()
        
        print("Took",timeend-timestart,"seconds to run",len(inputs),"samples.")

        # show(adv[0])
        # adv[0]
        # inputs[0]
        # x = ((adv[0] + 0.5) * 255)
        # ((adv[0] + 0.5) * 255).round().reshape((28,28))
        # plt.imshow(((adv[0] + 0.5) * 255).round().reshape((28,28)), cmap='Greys')
        # plt.imshow((adv[0] + 0.5) * 255)
        # plt.imshow(data[0])
        # plt.show()
        show_img(adv[8])

        for i in range(len(adv)):
            print("Valid:")
            show(inputs[i])
            print("Adversarial:")
            show(adv[i])
            
            print("Classification:", model.model.predict(adv[i:i+1]))

            print("Total distortion:", np.sum((adv[i]-inputs[i])**2)**.5)

def __test():
    with tf.Session() as sess:
        test_cw()

    with sess:
        test_cw()

def test_cw():
    sess = tf.Session()
    # sess.run(tf.global_variables_initializer())
    
    # keras maintains a tf session. It must be set by either
    # keras.backend.set_session(sess), or use inside a context manager
    # sess.as_default()
    with sess.as_default():
        data, model =  MNIST(), MNISTModel("models/mnist", sess)
    with sess.as_default():
        data, model =  CIFAR(), CIFARModel("models/cifar", sess)

    # testing the model
    np.argmax(model.model.predict(data.test_data[:10]), axis=1)
    print(np.argmax(data.test_labels[:10], axis=1))

    #data, model =  CIFAR(), CIFARModel("models/cifar", sess)
    attack_l2 = CarliniL2(sess, model, batch_size=10,
                          max_iterations=1000, confidence=0)
    attack_l0 = CarliniL0(sess, model, max_iterations=1000,
                          initial_const=10, largest_const=15)
    attack_li = CarliniLi(sess, model)

    inputs, targets = generate_data(data, samples=1, targeted=True,
                                    start=0, inception=False)
    # TODO find the first digits of each kind, try map it to the next digit
    inputs, targets = generate_data_2(data)

    adv_l2 = attack_l2.attack(inputs, targets)
    adv_l0 = attack_l0.attack(inputs, targets)
    adv_li = attack_li.attack(inputs, targets)

    plt.tight_layout()
    plt.tight_layout(pad=1, w_pad=1, h_pad=1)
    
    grid_show_image(inputs, 10, 1, 'images/orig-mnist.png')
    grid_show_image(adv_l2, 10, 1, 'images/l2.png')
    grid_show_image(adv_l0, 10, 1, 'images/l0.png')
    grid_show_image(adv_li, 9, 2, 'images/li.png')

    from contextlib import redirect_stdout
    redirect_stdout

    np.sum((adv_l2[0] - inputs[0]) ** 2)
    
    # np.argmax(targets, axis=1)
    # import keras
    # keras.backend.set_session(sess)
    np.argmax(model.model.predict(inputs), axis=1)
    np.argmax(targets, axis=1)
    # # (((adv_l2 + 0.5)*255).round())
    
    np.argmax(model.model.predict(adv_l2), axis=1)
    np.argmax(model.model.predict(adv_l0), axis=1)
    np.argmax(model.model.predict(adv_li), axis=1)
    
    np.sum(model.model.predict(adv_l2), axis=1)

    np.sum(sess.run(tf.nn.softmax(model.model.predict(adv_l2))), axis=1)

    softmax_pred = sess.run(tf.nn.softmax(model.model.predict(adv_l2)))
    softmax_pred[0]
    np.argmax(softmax_pred, axis=1)
    
    keras.activations.softmax(model.model)
    
    model.model.predict(((adv_l2 + 0.5)*255).round())
    # print(np.argmax(data.test_labels[:10], axis=1))

    
def show_img(img):
    """
    Show image by matplotlib.
    """
    img = np.round((img + 0.5) * 255).reshape((28,28))
    plt.imshow(img, cmap='Greys')
    plt.show()

def convert_image_255(img):
    return np.round(255 - (img + 0.5) * 255).reshape((28, 28))
    # return np.round(img).reshape((28, 28))
    # return np.round((img + 0.5) * 255).reshape((32, 32, 3))

def __test():
    adv
    sample_images(adv)
    grid_show_image(adv, 3, 3, '')

def sample_images(images):
    grid_shape = (3,3)
    # sample 20 imags
    indices = random.sample(range(len(images)), 20)
    images = images[indices]
    # show it
    grid_show_image(images, 10, 2)


def grid_show_image(images, width, height, filename='out.png'):
    """
    Sample 10 images, and save it.
    """
    assert len(images) == width * height
    plt.ioff()
    figure = plt.figure()
    # figure = plt.figure(figsize=(6.4, 1.2))
    figure.canvas.set_window_title('My Grid Visualization')
    for x in range(height):
        for y in range(width):
            # print(x,y)
            # figure.add_subplot(height, width, x*width + y + 1)
            ax = plt.subplot(height, width, x*width + y + 1)
            ax.set_axis_off()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # ax.set_visible(False)
            # plt.axis('off')
            # plt.imshow(images[x*width+y], cmap='gray')
            plt.imshow(convert_image_255(images[x*width+y]), cmap='gray')
            # plt.imshow(convert_image_255(images[x*width+y]))
            # plt.imshow(images[x*width+y])
    # plt.show()
    # plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    # plt.savefig(filename)
    return figure


def __test_cifar():
    grid_show_image(inputs, 10, 1, 'images/orig-mnist.png')

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    plt.imshow(x_train[9])
    y_train[0]
    plt.show()
    
    plt.imshow(convert_image_255(inputs[8]))
    img = convert_image_255(inputs[0]).reshape(3,32,32).transpose([1, 2, 0])

    data = data.reshape(data.shape[0], 3, 32, 32)
    final = np.zeros((data.shape[0], 32, 32, 3),dtype=np.float32)
    final[:,:,:,0] = data[:,0,:,:]
    final[:,:,:,1] = data[:,1,:,:]
    final[:,:,:,2] = data[:,2,:,:]

    img = convert_image_255(inputs[10])
    img_2 = np.zeros((3, 32, 32))
    img_2[0,:,:] = img[:,:,0]
    img_2[1,:,:] = img[:,:,1]
    img_2[2,:,:] = img[:,:,2]
    plt.imshow(img_2.transpose([1,2,0]))
    plt.imshow(img_2)
    plt.show()
    
    plt.imshow(img)
    plt.show()
