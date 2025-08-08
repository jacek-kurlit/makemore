import torch
import torch.nn.functional as F
import random


def build_dataset(words, blok_size):
    X, Y = [], []
    for w in words:
        context = [0] * blok_size
        for ch in w + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    return X, Y


def eval_loss(X, Y):
    emd = C[X]  # embed characters into vectors
    embcat = emd.view(
        emd.shape[0], -1
    )  # concatenate vectors so for example (N, 3, 2) -> (N, 6)
    hpreact = embcat @ W1 + b1  # hidden layer pre-activation
    h = torch.tanh(hpreact)  # hidden layer
    logits = h @ W2 + b2  # output layer

    return F.cross_entropy(logits, Y)


@torch.no_grad()  # tells PyTorch to skip book keeping track of gradients so it is faster
def split_loss(split):
    x, y = {"train": (Xtr, Ytr), "val": (Xdev, Ydev), "test": (Xte, Yte)}[split]
    loss = eval_loss(x, y)
    print(f"{split} loss: {loss.item():.4f}")


if __name__ == "__main__":
    words = open("names.txt", "r").read().splitlines()
    chars = sorted(list(set("".join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0  # add a special token for the end of the word
    itos = {i: s for s, i in stoi.items()}
    vocab_size = len(itos)
    blok_size = 3  # context length

    import random

    random.seed(42)  # for reproducibility
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))
    Xtr, Ytr = build_dataset(words[:n1], blok_size)  # 80% for training
    Xdev, Ydev = build_dataset(words[n1:n2], blok_size)  # 10% for validation
    Xte, Yte = build_dataset(words[n2:], blok_size)  # 10% for testing

    g = torch.Generator().manual_seed(2147483647)
    n_hidden = 200  # number of neurons in the hidden layer
    n_embd = 10  # the dimension of the embedding
    size = blok_size * n_embd  # size of the input vector
    C = torch.randn((vocab_size, n_embd), generator=g)
    # What is this (5/3) / (n_embd * blok_size) ** 0.5?
    # # It is a scaling factor for the weights to ensure that the initial loss is not too high.
    # It was found by some guys in Kaiming init paper: https://arxiv.org/abs/1502.01852
    W1 = (
        torch.randn(((size, n_hidden)), generator=g)
        * (5 / 3)
        / (n_embd * blok_size) ** 0.5
    )
    b1 = torch.randn(n_hidden, generator=g) * 0.01
    # We reduce initial loss.
    # Expected initial loss is around 3.0
    W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01
    b2 = torch.randn(vocab_size, generator=g) * 0
    parameters = [C, W1, b1, W2, b2]

    for p in parameters:
        p.requires_grad = True

    loss = torch.tensor(0.0)
    batch_size = 32
    max_steps = 100000  # number of training steps
    print("Training data")
    for i in range(max_steps):
        # minibatch construction
        ix = torch.randint(
            0, Xtr.shape[0], (batch_size,), generator=g
        )  # batch size of 32
        Xb, Yb = Xtr[ix], Ytr[ix]

        # forward pass
        loss = eval_loss(Xb, Yb)

        # backward pass
        for p in parameters:
            p.grad = None
        loss.backward()

        # update
        lr = 0.1 if i < 100000 else 0.01
        for p in parameters:
            p.data += -lr * p.grad

        if i % 10000 == 0:
            print(f"{i:7d}/{max_steps:7d}: {loss.item():.4f}")
        # tract stats if you crate plot for them you will find good learning rate, 0.1 is actually a good one
        # lri.append(lre[i])
        # lossi.append(loss.item())

    split_loss("train")
    split_loss("val")

    print("Sample from the model:")
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
