import numpy as np

def linear_forward(X, W, b):
    N = X.shape[0]
    M = W.shape[1]
    D = W.shape[0]

    X_reshape = np.reshape(X, (N, D))
    b = b.reshape((1, M))
    b_aug = np.repeat(b, N, axis=0)

    out = np.dot(X_reshape, W) + b_aug

    return out

def linear_backward(dout, X, W, b):
    N = X.shape[0]
    D = W.shape[0]

    X_reshape = np.reshape(X, (N, D))

    dW = np.dot(np.transpose(X_reshape), dout)

    dX = np.dot(dout, np.transpose((W))).reshape(X.shape)

    db = np.sum(dout, axis=0)

    return dX, dW, db

def relu_forward(X):

    out = X.copy()
    out[out < 0] = 0

    return out

def relu_backward(dout, X):
    dX = np.where(X > 0, dout, 0)
    return dX

def softmax(logits, y):
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))

    probs = probs / np.sum(probs, axis=1, keepdims=True)

    N = logits.shape[0]

    loss = -np.sum(np.log(probs[np.arange(N), y])) / N

    dlogits = probs.copy()

    dlogits[np.arange(N), y] -= 1

    dlogits = dlogits / N

    return loss, dlogits