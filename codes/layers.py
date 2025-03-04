import numpy as np


class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable
        self._saved_tensor = None

    def forward(self, input):
        pass

    def backward(self, grad_output):
        pass

    def update(self, config):
        pass

    def _saved_for_backward(self, tensor):
        '''The intermediate results computed during forward stage
        can be saved and reused for backward, for saving computation'''

        self._saved_tensor = tensor


class Selu(Layer):
    def __init__(self, name):
        super(Selu, self).__init__(name)

    def forward(self, input):
        # TODO START
        self._saved_for_backward(input)
        return 1.0507 * np.where(input > 0, input, 1.67326 * (np.exp(input) - 1))
        # TODO END

    def backward(self, grad_output):
        # TODO START
        input = self._saved_tensor
        return grad_output * 1.0507 * np.where(input > 0, 1, 1.67326 * np.exp(input))
        # TODO END

class HardSwish(Layer):
    def __init__(self, name):
        super(HardSwish, self).__init__(name)

    def forward(self, input):
        # TODO START
        self._saved_for_backward(input)
        return np.where(input <= -3, 0, np.where(input >= 3, input * (input + 3) / 6, input))
        # TODO END

    def backward(self, grad_output):
        # TODO START
        input = self._saved_tensor
        return grad_output * np.where(input <= -3, 0, np.where(input >= 3, (2 * input + 3) / 6, 1))
        # TODO END

class Tanh(Layer):
    def __init__(self, name):
        super(Tanh, self).__init__(name)

    def forward(self, input):
        # TODO START
        self._saved_for_backward(input)
        return (np.exp(input) - np.exp(-input)) / (np.exp(input) + np.exp(-input))
        # TODO END
    
    def backward(self, grad_output):
        # TODO START
        input = self._saved_tensor
        tmp = self.forward(input)
        return grad_output * (1 - tmp * tmp)
        # TODO END

class Softmax(Layer):
    def __init__(self, name):
        super(Softmax, self).__init__(name)

    def forward(self, input):
        self._saved_for_backward(input)
        tmp = np.exp(input)
        return tmp / np.sum(tmp, axis = 1, keepdims = True)
    
    def backward(self, grad_output):
        input = self._saved_tensor
        softmax_output = self.forward(input)
        grad_input = np.zeros_like(input)
        
        for i in range(input.shape[0]):
            s = softmax_output[i].reshape(-1, 1)
            jacobian_matrix = np.diagflat(s) - np.dot(s, s.T)
            grad_input[i] = np.dot(jacobian_matrix, grad_output[i])
        
        return grad_input

class Linear(Layer):
    def __init__(self, name, in_num, out_num, init_std):
        super(Linear, self).__init__(name, trainable=True)
        self.in_num = in_num
        self.out_num = out_num
        self.W = np.random.randn(in_num, out_num) * init_std
        self.b = np.zeros(out_num)

        self.grad_W = np.zeros((in_num, out_num))
        self.grad_b = np.zeros(out_num)

        self.diff_W = np.zeros((in_num, out_num))
        self.diff_b = np.zeros(out_num)

    def forward(self, input):
        # TODO START
        self._saved_for_backward(input)
        return np.dot(input, self.W) + self.b
        # TODO END

    def backward(self, grad_output):
        # TODO START
        input = self._saved_tensor
        self.grad_W = np.dot(input.T, grad_output)
        self.grad_b = np.sum(grad_output, axis=0)
        return np.dot(grad_output, self.W.T)
        # TODO END

    def update(self, config):
        mm = config['momentum']
        lr = config['learning_rate']
        wd = config['weight_decay']

        self.diff_W = mm * self.diff_W + (self.grad_W + wd * self.W)
        self.W = self.W - lr * self.diff_W

        self.diff_b = mm * self.diff_b + (self.grad_b + wd * self.b)
        self.b = self.b - lr * self.diff_b
