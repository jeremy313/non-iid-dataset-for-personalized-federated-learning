# non-iid-dataset-for-personalized-federated-learning

This repository is the official implementation of the non-iid dataset in "LotteryFL: Personalized and Communication-Efficient Federated Learning with Lottery Ticket Hypothesis on Non-IID Datasets".

## Requirements
This implementation is based on torchvision, so although pytorch is not necessary for this dataset, you still need to install pytorch to utilize this.
```
torchvision = 0.4.0
numpy = 1.17.2
```

You still need to download mnist and cifar10 and put them under ./data/cifar and ./data/mnist

## Arguments
The two functions `get_dataset_cifar10_extr_noniid` and `get_dataset_mnist_extr_noniid` have the same four arguments:
```
num_users:      the number of clients you want to distributed data to
nclass:         the number of image classes each client has
nsamples:       number of samples per class distributed to clients
rate_unbalance: unbalanced rate of non-iid MNIST and CIFAR10 dataset
```

