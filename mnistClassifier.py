from layers import *

class MnistClassifier():
    def __init__(self, hidden_dims, input_size=28*28, num_of_classes=10,
                 reg=0.0, weight_scale=1e-2, dtype=np.float32, lr=0.002,
                 batch_size = 128, num_of_epochs=10, lr_decay=0.99, verbose_every=10, printable=True):

        # add number of class to the end of hidden_dim
        if len(hidden_dims) == 0:
            raise Exception('Cannot create a neural network with no hidden layers.')

        self.hidden_dims = hidden_dims.append(num_of_classes)
        self.num_of_classes = num_of_classes
        self.reg = reg
        self.num_of_layers = len(hidden_dims)
        self.dtype = dtype
        self.weight_scale = weight_scale
        self.params = {}
        self.printable = printable
        self.lr = lr
        self.batch_size = batch_size
        self.num_of_epoches = num_of_epochs
        self.verbose_every = verbose_every
        self.lr_decay = lr_decay

        cur_input_size = input_size

        for i in range(self.num_of_layers):
            self.params['w%d'%i], self.params['b%d'%i] = \
                self.params_initializer(cur_input_size, hidden_dims[i],
                                       self.weight_scale)
            cur_input_size = hidden_dims[i]

        if self.printable:
            for k, v in self.params.items():
                print (k+' is initialized with shape : ' + str(v.shape))

        # train configurations
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_accuracy_his = []
        self.val_accurracy_his = []

    def forward(self, X):
        X = X.astype(self.dtype)
        self.linear_cache, self.relu_cache = {}, {}

        self.linear_cache['l0'] = linear_forward(X, self.params['w0'], self.params['b0'])
        self.relu_cache['r0'] = relu_forward(self.linear_cache['l0'])

        for i in range(1, self.num_of_layers-1):
            self.linear_cache['l%d'%i] = linear_forward(self.relu_cache['r%d'%(i-1)],
                                                        self.params['w%d'%i], self.params['b%d'%i])
            self.relu_cache['r%d'%i] = relu_forward(self.linear_cache['l%d'%i])

        self.linear_cache['l%d'%(self.num_of_layers-1)] = linear_forward(
            self.relu_cache['r%d'%(self.num_of_layers-2)], self.params['w%d'%(self.num_of_layers-1)],
            self.params['b%d'%(self.num_of_layers-1)]
        )

        scores = self.linear_cache['l%d'%(self.num_of_layers-1)]

        return scores

    def loss(self, X, y=None):

        scores = self.forward(X)

        if y is None:
            return scores

        loss, grads, up_stream_grad = 0, {}, {}
        loss, up_stream_grad['dlogits'] = softmax(scores, y)

        if self.reg != 0:
            for i in range(self.num_of_layers):
                loss += 0.5*self.reg*np.sum(np.sum(self.params['w%d'%i]**2))

        up_stream_grad['dX_l%d'%(self.num_of_layers-1)], \
        grads['w%d'%(self.num_of_layers-1)], \
        grads['b%d'%(self.num_of_layers-1)] = \
            linear_backward(up_stream_grad['dlogits'], self.relu_cache['r%d'%(self.num_of_layers-2)],
                            self.params['w%d'%(self.num_of_layers-1)], self.params['b%d'%(self.num_of_layers-1)])

        grads['w%d'%(self.num_of_layers-1)] += self.reg*self.params['w%d'%(self.num_of_layers-1)]

        for i in range(self.num_of_layers-2, 0, -1):
            up_stream_grad['dX_r%d'%i] = relu_backward(
                up_stream_grad['dX_l%d'%(i+1)], self.linear_cache['l%d'%i])
            up_stream_grad['dX_l%d'%i], grads['w%d'%i], grads['b%d'%i] = \
                linear_backward(up_stream_grad['dX_r%d'%i], self.relu_cache['r%d'%(i-1)],
                                self.params['w%d'%i], self.params['b%d'%i])

            grads['w%d'%i] += self.reg*self.params['w%d'%(i)]

        up_stream_grad["dX_r0"] = \
            relu_backward(up_stream_grad["dX_l1"], self.linear_cache["l0"])

        up_stream_grad["dX_l0"], grads["w0"], grads["b0"] = \
            linear_backward(up_stream_grad["dX_r0"], X,
                            self.params["w0"], self.params["b0"])

        grads["w0"] = grads["w0"] + self.reg * self.params["w0"]

        return loss, grads

    def _step(self, X_batch, y_batch):
        loss, grads = self.loss(X_batch, y_batch)
        self.loss_history.append(loss)

        # update each params
        for k, v in self.params.items():
            self.params[k] -= self.lr*grads[k]

    def inference(self, X):
        N = X.shape[0]
        num_of_batches = N//self.batch_size + 1
        y_pred = []

        for i in range(num_of_batches):
            X_batch = X[i*self.batch_size:(i+1)*self.batch_size]
            scores = self.loss(X_batch)
            y_pred.append(np.argmax(scores, axis=1))

        y_pred = np.hstack(y_pred)

        return y_pred

    def cal_accuracy(self, X, y):

        y_pred = self.inference(X)

        return np.mean(y_pred == y)


    def params_initializer(self, n_in, n_out, weight_scale=1e-2):
        w = weight_scale*np.random.randn(n_in, n_out)
        b = np.random.randn(n_out, )

        return w, b

    def train(self, X, y):

        X_train = X['train']
        X_val = X['val']
        y_train = y['train']
        y_val = y['val']

        num_of_data = X_train.shape[0]
        num_of_batches = num_of_data//self.batch_size + 1

        for i in range(self.num_of_epoches):

            for itr in range(num_of_batches):
                X_batch = X_train[itr*self.batch_size:(itr+1)*self.batch_size]
                y_batch = y_train[itr*self.batch_size:(itr+1)*self.batch_size]
                self._step(X_batch, y_batch)
                if itr % self.verbose_every == 0:
                    log = '[Epoch %3d/%3d] Iteration: %5d/%5d, loss: %8f' % \
                          (i+1, self.num_of_epoches, itr, num_of_batches, self.loss_history[-1])
                    print (log)
            self.lr = self.lr * self.lr_decay

            train_accuracy = self.cal_accuracy(X_train, y_train)
            self.train_accuracy_his.append(train_accuracy)
            val_accuracy = self.cal_accuracy(X_val, y_val)
            self.val_accurracy_his.append(val_accuracy)

            print ('[Epoch %3d/%3d : train accuracy: %4f, validation accuracy: %4f]'
                   % (i+1, self.num_of_epoches, train_accuracy, val_accuracy))

            if val_accuracy > self.best_val_acc:
                self.best_val_acc = val_accuracy
                self.best_params = {}
                for k, v in self.params.items():
                    self.best_params[k] = v.copy()





