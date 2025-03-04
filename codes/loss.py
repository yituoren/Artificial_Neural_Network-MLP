from __future__ import division
import numpy as np


class KLDivLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        tmp = np.exp(input)
        tmp = tmp / np.sum(tmp, axis = 1, keepdims = True)
        tmp = np.sum(np.where(target > 0, target * (np.log(target) - np.log(tmp)), 0), axis = 1)
        return np.sum(tmp, axis = 0) / input.shape[0]
        # TODO END

    def backward(self, input, target):
		# TODO START
        tmp = np.exp(input)
        tmp = tmp / np.sum(tmp, axis = 1, keepdims = True)
        return (tmp - target) / input.shape[0]
		# TODO END


class SoftmaxCrossEntropyLoss(object):
    def __init__(self, name):
        self.name = name

    def forward(self, input, target):
        # TODO START
        tmp = np.exp(input)
        tmp = tmp / np.sum(tmp, axis = 1, keepdims = True)
        tmp = -np.sum(target * np.log(tmp), axis = 1)
        return np.sum(tmp, axis = 0) / input.shape[0]
        # TODO END

    def backward(self, input, target):
        # TODO START
        tmp = np.exp(input)
        tmp = tmp / np.sum(tmp, axis = 1, keepdims = True)
        return (tmp - target) / input.shape[0]
        # TODO END


class HingeLoss(object):
    def __init__(self, name, margin=0.5):
        self.name = name
        self.margin = margin

    def forward(self, input, target):
        # TODO START 
        tmp = np.where(target > 0, 0, np.maximum(self.margin + input - np.sum(np.where(target > 0, input, 0), axis = 1, keepdims = True), 0))
        return np.mean(np.sum(tmp, axis = 1))
        # TODO END

    def backward(self, input, target):
        # TODO START
        tmp = np.where((self.margin + input - np.sum(np.where(target > 0, input, 0), axis = 1, keepdims = True) > 0) & (target < 1), 1, 0)
        tmp = np.where(target > 0, -np.sum(tmp, axis = 1, keepdims = True), tmp)
        return tmp / input.shape[0]
        # TODO END


# Bonus
class FocalLoss(object):
    def __init__(self, name, alpha=None, gamma=2.0):
        self.name = name
        if alpha is None:
            self.alpha = [0.1 for _ in range(10)]
        self.gamma = gamma

    def forward(self, input, target):
        # TODO START
        tmp = np.exp(input)
        tmp = tmp / np.sum(tmp, axis = 1, keepdims = True)
        re_alpha = np.array(self.alpha).reshape(1, -1)
        tmp = -(re_alpha * target + (1 - re_alpha) * (1 - target)) * ((1 - tmp) ** self.gamma) * target * np.log(tmp)
        return np.mean(np.sum(tmp, axis = 1))
        # TODO END

    def backward(self, input, target):
        # TODO START
        softmax_output = np.exp(input)
        softmax_output = softmax_output / np.sum(softmax_output, axis = 1, keepdims = True)

        re_alpha = np.array(self.alpha).reshape(1, -1)
        grad_output = ((1 - softmax_output) ** self.gamma) / softmax_output - self.gamma * ((1 - softmax_output) ** (self.gamma - 1)) * np.log(softmax_output)
        grad_output = -(re_alpha * target + (1 - re_alpha) * (1 - target)) * target * grad_output

        grad_input = np.zeros_like(input)        
        for i in range(input.shape[0]):
            s = softmax_output[i].reshape(-1, 1)
            jacobian_matrix = np.diagflat(s) - np.dot(s, s.T)
            grad_input[i] = np.dot(jacobian_matrix, grad_output[i])

        return grad_input / input.shape[0]
        # TODO END