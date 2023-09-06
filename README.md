# PyTorch-Classification

This repository contains several Jupyter notebooks which explore classification tasks with PyTorch. We will build a convolutional neural network (CNN) to detect smiling in images from the CelebA dataset, which contains over 200000 celebrity photographs, and also study a variety of methods for recognizing hand-drawn digits from the MNIST dataset. 

Initially, we recap the fundamentals of PyTorch in the notebook <b>PyTorch_Fundamentals.ipynb</b>. Here we review basic tensor operations in PyTorch and building neural networks, mostly following Chapters 12 and 13 of <i>''Machine Learning with PyTorch and Scikit-Learn''</i> by Sebastian Raschka et al.

Before tackling MNIST digit recognition with neural networks, we first explore the performance of several learning algorithms such as K-Nearest Neighbors and Random Forest on this task, as shown in the notebook <b>MNIST_scikit-learn.ipynb</b>.

Then we build a neural network (NN) in PyTorch and train it to classify MNIST images in the notebook <b>MNIST_PyTorch-NN.ipynb</b>. First we construct an inital model, using two hidden layers followed by a $10$-unit output layer with softmax. We perform hyperparameter tuning, and decide on good values for the learning rate $\alpha$, number of hidden units, weight decay (L2 penalty), and dropout probability $p$. We see how implementing these variance reduction techniques improves test-set performance compared to an initial baseline model.

After this, we construct a CNN in PyTorch and train it on the MNIST dataset, shown in <b>MNIST_PyTorch_CNN.ipynb</b>. We compare its performance to our previous neural network model, and find a marked improvement, with test accuracy now exceeding 99%.

Finally, in the notebook <b>CelebA_Smile_Detection.ipynb</b>, we explore the CelebA dataset and train a CNN in PyTorch to detect smiles in these images. Here we will employ a range of image augmentation techniques using the torchvision package.
