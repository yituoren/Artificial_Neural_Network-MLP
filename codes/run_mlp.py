from network import Network
from utils import LOG_INFO
from layers import Selu, HardSwish, Linear, Tanh, Softmax
from loss import KLDivLoss, SoftmaxCrossEntropyLoss, HingeLoss, FocalLoss
from solve_net import train_net, test_net
from load_data import load_mnist_2d
import matplotlib.pyplot as plt


train_data, test_data, train_label, test_label = load_mnist_2d('data')

# Your model defintion here
# You should explore different model architecture
model = Network()
model.add(Linear('fc1', 784, 112, 0.01))
model.add(Tanh('af1'))
model.add(Linear('fc2', 112, 28, 0.01))
model.add(Tanh('af2'))
model.add(Linear('fc3', 28, 10, 0.01))

loss = FocalLoss(name='loss')

# Training configuration
# You should adjust these hyperparameters
# NOTE: one iteration means model forward-backwards one batch of samples.
#       one epoch means model has gone through all the training samples.
#       'disp_freq' denotes number of iterations in one epoch to display information.

config = {
    'learning_rate': 0.03,
    'weight_decay': 0.0001,
    'momentum': 0.9,
    'batch_size': 100,
    'max_epoch': 100,
    'disp_freq': 50,
    'test_epoch': 5
}

train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(config['max_epoch']):
    LOG_INFO('Training @ %d epoch...' % (epoch))
    tmp_train_loss, tmp_train_acc = train_net(model, loss, config, train_data, train_label, config['batch_size'], config['disp_freq'])

    train_loss += tmp_train_loss
    train_acc += tmp_train_acc

    if epoch % config['test_epoch'] == 0:
        LOG_INFO('Testing @ %d epoch...' % (epoch))
        tmp_test_loss, tmp_test_acc = test_net(model, loss, test_data, test_label, config['batch_size'])

        test_loss.append(tmp_test_loss)
        test_acc.append(tmp_test_acc)


plt.figure()

plt.subplot(2, 2, 1)
plt.plot(train_loss, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(train_acc, label='Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(test_loss, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Test Loss')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(test_acc, label='Accuracy')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.legend()

plt.show()