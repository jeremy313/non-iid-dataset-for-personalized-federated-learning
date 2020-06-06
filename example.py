from dataset.cifar10_noniid import get_dataset_cifar10_extr_noniid
from dataset.mnist_noniid import get_dataset_mnist_extr_noniid

num_users_cifar = 400
nclass_cifar = 2
nsamples_cifar = 20
rate_unbalance_cifar = 1.0

num_users_mnist = 400
nclass_mnist = 2
nsamples_mnist = 20
rate_unbalance_mnist = 1.0



train_dataset_cifar, test_dataset_cifar, user_groups_train_cifar, user_groups_test_cifar = get_dataset_cifar10_extr_noniid(num_users_cifar, nclass_cifar, nsamples_cifar, rate_unbalance_cifar)

train_dataset_mnist, test_dataset_mnist, user_groups_train_mnist, user_groups_test_mnist = get_dataset_mnist_extr_noniid(num_users_mnist, nclass_mnist, nsamples_mnist, rate_unbalance_mnist)


