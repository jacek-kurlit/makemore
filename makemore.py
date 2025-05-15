from typing import Tuple
import torch

# this is special token for start and end sequence
SEQUENCE_TOKEN = "."

# this is random generator with seed so outcomes are deterministic
GENERATOR = torch.Generator().manual_seed(2147483647)


# this does not work in terminal...
def print_biagram(N, itos):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap="Blues")
    for i in range(27):
        for j in range(27):
            chstr = itos[i] + itos[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color="gray")
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color="gray")
    plt.axis("off")


# this is just rust equivalent of window of 2: emma -> .e, em, mm, ma, a.
def as_word_window(word):
    chs = [SEQUENCE_TOKEN] + list(word) + [SEQUENCE_TOKEN]
    return zip(chs, chs[1:])


def measure_learning_efficiency(words, P, stoi):
    print("measuring learning efficiency")
    # log makes good loss function
    # generally we want to know how good/bad model is
    # If log_likelihood is near 0 it means it is good at prediction training set
    log_likelihood = 0.0
    n = 0
    for w in words:
        for ch1, ch2 in as_word_window(w):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            prob = P[ix1, ix2]
            # log(prob) will return 0 if prob == 1, and -inf for prob == 0
            logprob = torch.log(prob)
            log_likelihood += logprob
            n += 1
            # print(f"{ch1}{ch2}: {prob:.4f} {logprob:.4f}")
    print(f"{log_likelihood=}")
    # we just make it positive so 0 means good +inf bad
    nll = -log_likelihood
    print(f"{nll=}")
    # usually normalized values is used as loss function so we get average of log_likelihood
    print(f"{nll / n}")


def make_more(W, n, itos):
    print(f"Generating {n} next values")
    g = torch.Generator().manual_seed(2147483647)
    for _ in range(n):
        out = []
        ix = 0
        while True:
            # generally we are converting single char to input here
            xenc = torch.nn.functional.one_hot(
                torch.tensor([ix]), num_classes=27
            ).float()
            logits = xenc @ W
            counts = logits.exp()
            p = counts / counts.sum(1, keepdims=True)
            ix = torch.multinomial(
                p, num_samples=1, replacement=True, generator=g
            ).item()
            out.append(itos[ix])
            # 0 means end of sequence char
            if ix == 0:
                break
        print("".join(out))


# this is not NN yet but some basic statistic "model"
# it teaches what is going on pure probability ground
def biagram_statistic_model(words, itos, stoi):
    # this is tensor, you can imagine this as 2 dimensional matrix
    # rows and columns are chars indexes so for example 0 - '.' 1 - 'a',..., 27 - 'z'
    # value at [0,0] means how many times '.' was followed by '.'
    # [0,1] means how many times '.' was followed by 'a'
    # [27,27] means how many times 'z' was followed by 'z' etc.
    N = torch.zeros((27, 27), dtype=torch.int32)
    for w in words:
        for ch1, ch2 in as_word_window(w):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1
    # we do we add 1? To do model smoothing, we just do want to get prob of 0 because that would make log function to return -inf
    # we just add one so unlikely names like 'adrejq' are still possible but rather of small chances
    P = (N + 1).float()
    # we want to have sum across rows thus 1(0 means sum of columns)
    # this normalizes counts, it's now probability so chars that occur often will be close to 1 and others to 0
    # /= may be faster than x = x /... because memory allocation
    P /= P.sum(1, keepdim=True)
    for _ in range(5):
        out = []
        ix = 0
        while True:
            p = P[ix]
            ix = torch.multinomial(
                p, num_samples=1, replacement=True, generator=GENERATOR
            ).item()
            out.append(itos[ix])
            # 0 means end of sequence char
            if ix == 0:
                break
        print("".join(out))
    measure_learning_efficiency(words, P, stoi)


def some_exaple_explanation(xs, ys, itos, probs):
    # this is 'emma' example
    nlls = torch.zeros(5)
    for i in range(5):
        x = xs[i].item()  # input char index
        y = ys[i].item()  # label char index
        print("--------------------")
        print(f"biagram example {i + 1}: {itos[x]}{itos[y]} (indexes {x}, {y})")
        print(f"input to the natural net: {x}")
        print(f"output probability from net: {probs[i]}")
        print(f"label (actual next char): {y}")
        p = probs[i, y]
        print(f"probability assinged by net to the corrcet char: {p.item()}")
        # this should be close to 0 to tell that net is doing good work
        logp = torch.log(p)
        print(f"log likelihood: {logp.item()}")
        nll = -logp
        print(f"negative log likelihood: {nll.item()}")
        nlls[i] = nll
    print("--------------------")
    # this tell how net is doing in general, it should be close to 0
    print(f"average negative log likelihood: {nlls.mean().item()}")


def create_traning_set(words, stoi) -> Tuple[torch.Tensor, torch.Tensor]:
    print("Creating training set")
    xs, ys = [], []
    for w in words:
        for ch1, ch2 in as_word_window(w):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)
    xs = torch.tensor(xs)  # input chars indexes
    ys = torch.tensor(ys)  # label chars indexes
    return (xs, ys)


def forward_pass(xs, ys, W) -> torch.Tensor:
    # this is 27 for our training data, simply 26 latin chars + '.'
    num = xs.nelement()
    # NN expects float vales not integers
    # What this function does?
    # it takes xs [5, 13, 13, 1, 0] ("emma.") and converts it to vector 5x27 where (5 chars in xs and 27 total chars)
    # v[0][0] = 1, v[1][5] = 1, v[2][13] = 1, v[3][13] = 1, v[4][1] = 1 and all other values are 0
    # we have 27 chars so we need 27 values
    xenc = torch.nn.functional.one_hot(xs, num_classes=27).float()
    # First we multiply matrices
    logits = xenc @ W  # log counts
    # Since this numbers may be negative we use exp function to turn them into range (0, +inf)
    # exp(0) = 1 so anything below was negative anything bigger was a positive
    counts = logits.exp()  # equivalent to N matrix from biagram_statistic_model - the matrix with counts(how many times 'e' was followed by 'm')
    # probs[0] is input 0 probability (for '.emma' this is probability for '.' so probs[0][5] will hold how probable is to for 'e' to follow '.')
    probs = counts / counts.sum(1, keepdims=True)
    # btw last 2 lines are just soft max function
    #
    # This is trick, we want weights to be close to 0 because non 0 weights increase overall loss
    # this optimize W
    regularization_loss = 0.01 * (W**2).mean()
    # arange gives vec [0,1,2,3,4]
    # let's consider 'emma.', ys == [5,13,13,1,0]
    # this is simply is [probs[0, 5], probs[1, 13]...]
    # -log().mean() gives us average negative log_likelihood which is our loss function
    loss = -probs[torch.arange(num), ys].log().mean() + regularization_loss
    return loss


LEARNING_RATE = -50.0
TRANING_INTERATIONS = 400


def backward_pass(W, loss):
    W.grad = None
    loss.backward()
    W.data += LEARNING_RATE * W.grad


def biagram_natural_network_model(words, itos, stoi):
    (xs, ys) = create_traning_set(words, stoi)
    # this are weights of our model, we begin with random values
    W = torch.randn((27, 27), generator=GENERATOR, requires_grad=True)
    print("Training network")
    for i in range(TRANING_INTERATIONS):
        loss = forward_pass(xs, ys, W)
        print(f"{i} - current loss {loss}")
        backward_pass(W, loss)
    make_more(W, 5, itos)
    # some_exaple_explanation(xs, ys, itos, probs)


if __name__ == "__main__":
    words = open("names.txt").read().splitlines()
    # this simply returns all unique chars
    chars = sorted(list(set("".join(words))))
    # this is a map of char to index to a is 1 z is 26 basically
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi[SEQUENCE_TOKEN] = 0
    itos = {i: s for s, i in stoi.items()}
    # biagram_statistic_model(words, itos, stoi)
    biagram_natural_network_model(words, itos, stoi)
