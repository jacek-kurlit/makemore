import torch
import torch.nn.functional as F

if __name__ == "__main__":
    words = open("names.txt", "r").read().splitlines()
    chars = sorted(list(set("".join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0  # add a special token for the end of the word
    itos = {i: s for s, i in stoi.items()}
    blok_size = 3  # context length
    g = torch.Generator().manual_seed(2147483647)  # seed for reproducibility
    X, Y = [], []
    # number of neurons in the hidden layer
    neurons_size = 100
    # 27 is the number of possible characters
    possible_chars_size = len(stoi)
    size = blok_size * 2  # size of the input vector
    C = torch.randn((possible_chars_size, 2), generator=g)
    W1 = torch.randn(((size, neurons_size)), generator=g)
    b1 = torch.randn(neurons_size, generator=g)
    W2 = torch.randn((neurons_size, possible_chars_size), generator=g)
    b2 = torch.randn(possible_chars_size, generator=g)
    parameters = [C, W1, b1, W2, b2]
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

    for p in parameters:
        p.requires_grad = True

    for _ in range(1000):
        # minibatches
        # We want to speed up training by using minibatches, quality is lower but training is faster
        ix = torch.randint(0, X.shape[0], (32,), generator=g)  # batch size of 32
        emd = C[X[ix]]
        ## view turns (N, 3 , 2) -> (N, 6) in very efficient way
        h = torch.tanh(emd.view(emd.shape[0], size) @ W1 + b1)  # (N, neurons_size)
        logits = h @ W2 + b2

        # Loss can be computed using cross entropy, they are equivalent to (but more efficient):
        # counts = logits.exp()
        # prob = counts / counts.sum(dim=1, keepdim=True)  # (N, possible_chars_size)
        # loss = -prob[torch.arange(prob.shape[0]), Y].log().mean()
        loss = F.cross_entropy(logits, Y[ix])
        print(loss.item())
        for p in parameters:
            p.grad = None

        loss.backward()

        for p in parameters:
            # 0.1 is the learning rate
            p.data += -0.1 * p.grad
