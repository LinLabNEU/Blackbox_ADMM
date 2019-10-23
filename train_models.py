

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

import tensorflow as tf
from setup_mnist import MNIST
from setup_cifar import CIFAR
import os


def train(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1, init=None,
          adversarial=False, examples=None, labels=None):
    """
    Standard neural network training procedure.
    """
    if adversarial:
        data.train_data = np.concatenate((data.train_data, examples), axis=0)
        data.train_labels = np.concatenate((data.train_labels, labels), axis=0)
    model = Sequential()

    print(data.train_data.shape)
    
    model.add(Conv2D(params[0], (3, 3),
                            input_shape=data.train_data.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(params[1], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(params[2], (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(params[3], (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(params[4]))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(params[5]))
    model.add(Activation('relu'))
    model.add(Dense(10))
    
    if init != None:
        model.load_weights(init)

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    
    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              nb_epoch=num_epochs,
              shuffle=True)
    

    if file_name != None:
        model.save(file_name)

    return model

    
def main(args):
    if not os.path.isdir('models'):
        os.makedirs('models')


    if args['dataset'] == "mnist" or args['dataset'] == "all":
            train(MNIST(), "models/mnist", [32, 32, 64, 64, 200, 200], num_epochs=50)
    if args['dataset'] == 'cifar' or args['dataset'] == 'all':
            train(CIFAR(), "models/cifar", [64, 64, 128, 128, 256, 256], num_epochs=50)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", choices=["mnist", "cifar", "all"], default="all")
    parser.add_argument("-a", "--adversarial", action='store_true', default=False)
    parser.add_argument("-dd", "--defensive", action='store_true', default=False)
    parser.add_argument("-t", "--temp", nargs='+', type=int, default=0)
    args = vars(parser.parse_args())
    print(args)
    main(args)
