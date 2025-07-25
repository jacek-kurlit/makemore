import torch
import torch.nn.functional as F


def build_dataset(words, blok_size):
    X, Y = [], []
    for w in words:
        context = [0] * blok_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            # print("".join(itos[i] for i in context), "->", itos[ix])
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


def eval_loss(X, Y, W1, W2, b1, b2, C, size):
    emd = C[X]
    ## view turns (N, 3 , 2) -> (N, 6) in very efficient way
    h = torch.tanh(emd.view(emd.shape[0], size) @ W1 + b1)  # (N, neurons_size)
    logits = h @ W2 + b2

    # Loss can be computed using cross entropy, they are equivalent to (but more efficient):
    # counts = logits.exp()
    # prob = counts / counts.sum(dim=1, keepdim=True)  # (N, possible_chars_size)
    # loss = -prob[torch.arange(prob.shape[0]), Y].log().mean()
    return F.cross_entropy(logits, Y)


if __name__ == "__main__":
    words = open("names.txt", "r").read().splitlines()
    chars = sorted(list(set("".join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0  # add a special token for the end of the word
    itos = {i: s for s, i in stoi.items()}
    blok_size = 3  # context length
    g = torch.Generator().manual_seed(2147483647)  # seed for reproducibility
    Xtr, Ytr = [], []
    # number of neurons in the hidden layer
    neurons_size = 200
    # 27 is the number of possible characters
    possible_chars_size = len(stoi)
    embedding_size = 10
    size = blok_size * embedding_size  # size of the input vector
    C = torch.randn((possible_chars_size, embedding_size), generator=g)
    W1 = torch.randn(((size, neurons_size)), generator=g)
    b1 = torch.randn(neurons_size, generator=g)
    W2 = torch.randn((neurons_size, possible_chars_size), generator=g)
    b2 = torch.randn(possible_chars_size, generator=g)
    parameters = [C, W1, b1, W2, b2]
    import random

    random.seed(42)  # for reproducibility
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    Xtr, Ytr = build_dataset(words[:n1], blok_size)  # 80% for training
    Xdev, Ydev = build_dataset(words[n1:n2], blok_size)  # 10% for validation
    Xte, Yte = build_dataset(words[n2:], blok_size)  # 10% for testing

    for p in parameters:
        p.requires_grad = True

    # something between -1 and 0
    # initial values were eyeballed
    # lre = torch.linspace(-3, 0, 1000)
    # learning rates this in simply from 10^-3 to 10^0 , we are using exponential step here instead of linear
    # lrs = 10**lre  # learning rates
    # lri = []
    # lossi = []
    loss = torch.tensor(0.0)
    print("Training data")
    for i in range(200000):
        # minibatches
        # We want to speed up training by using minibatches, quality is lower bu
        ix = torch.randint(0, Xtr.shape[0], (32,))  # batch size of 32
        loss = eval_loss(Xtr[ix], Ytr[ix], W1, W2, b1, b2, C, size)

        for p in parameters:
            p.grad = None

        loss.backward()

        # lr = lrs[i]
        lr = 0.1
        if i > 100000:
            lr = 0.01
        for p in parameters:
            # 0.1 is the learning rate
            p.data += -lr * p.grad

        # tract stats if you crate plot for them you will find good learning rate, 0.1 is actually a good one
        # lri.append(lre[i])
        # lossi.append(loss.item())

    print("Training loss:", loss.item())
    # eval on the training set
    loss = eval_loss(Xdev, Ydev, W1, W2, b1, b2, C, size)
    print("Validation loss:", loss.item())

    print("Infering from the model:")
    # infer model
    for _ in range(20):
        out = []
        context = [0] * blok_size  # start with a context of zeros
        while True:
            emd = C[torch.tensor([context])]
            h = torch.tanh(emd.view(1, -1) @ W1 + b1)  # (N, neurons_size)
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]  # update context
            out.append(ix)
            if ix == 0:
                break
        print("".join(itos[i] for i in out))

