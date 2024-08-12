"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch


def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        x1 = self.layer1.forward(x).relu()
        x2 = self.layer2.forward(x1).relu()
        return self.layer3.forward(x2).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

    def forward(self, x):
        """
        Explanation for my future self.
        To forward x since we doesn't implement matrix multiplication yet
        we will need to do some Gimmicks with shape/dimension.
        1- We know that the last dim of x would correspond to the first dim
        of weights.
        x = (a, b, c,....d)
        b =          (d, e)
        We want the d dimension to be aligned so that we can broadcasting the shapes
        hence we add 1 to x.
        x = (a, b, c,....d, 1)
        b =             (d, e)
        After the element wise multiplication we will get an output of shape
        x = (a, b, c,....d, e)
        Now we sum over the d dimension and get an output of shape
        x = (a, b, c,....1, e)
        We finally squeeze that inner 1 dimension to get the final output dim
        x = (a, b, c,....., e)
        """
        # add an extra dimension to allow broadcasting.
        x = x.view(*x.shape, 1)

        # element wise multiplication.
        x = x * self.weights.value

        # Sum over the second to last dimension. (Need to figure out WHY. TODO)
        x = x.sum(len(x.shape) - 2)

        # Reshape by removing the second to last dimension.
        x = x.view(*(x.shape[:-2] + x.shape[-1:]))
        # x = x.view(*(x.shape[:len(x.shape) - 2] + x.shape[len(x.shape) - 1:]))

        # add the bias
        x = x + self.bias.value

        # In matrix wise (return x * self.weights.value + self.bias.value
        return x


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)
