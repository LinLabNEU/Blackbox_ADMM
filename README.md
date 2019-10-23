# Blackbox_ADMM


Implementations of the black-box ADMM attack algorithms in Tensorflow. It runs correctly
on Python 3.6.

To evaluate the robustness of a neural network, create a model class with a
predict method that will run the prediction network *without softmax*. 

### Pre-requisites

The following steps should be sufficient to get these attacks up and running on
most Linux-based systems.

```bash
    sudo apt-get install python3-pip
    sudo pip3 install --upgrade pip
    sudo pip3 install pillow scipy numpy tensorflow-gpu keras h5py
```

#### To create the MNIST/CIFAR models:

```bash
python3 train_models.py
```

#### To download the inception model:

```bash
python3 setup_inception.py
```

#### And finally to test the attacks

```bash
python3 test_attack.py
```
