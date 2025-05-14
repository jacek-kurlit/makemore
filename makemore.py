import torch

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


def measure_learning_efficiency(words, P, stoi):
    print("measuring learning efficiency")
    # log makes good loss function
    # generally we want to know how good/bad model is
    # If log_likelihood is near 0 it means it is good at prediction training set
    log_likelihood = 0.0
    n = 0
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
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


# this is not NN yet but some basic statistic "model"
# it teaches what is going on pure probability ground
def biagram_statistic_model(words):
    # this simply returns all unique chars
    chars = sorted(list(set("".join(words))))
    # this is a map of char to index to a is 1 z is 26 basically
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    # this is special token for start and end sequence
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}
    # this is tensor, you can imagine this as 2 dimensional matrix
    # rows and columns are chars indexes so for example 0 - '.' 1 - 'a',..., 27 - 'z'
    # value at [0,0] means how many times '.' was followed by '.'
    # [0,1] means how many times '.' was followed by 'a'
    # [27,27] means how many times 'z' was followed by 'z' etc.
    N = torch.zeros((27, 27), dtype=torch.int32)
    for w in words:
        chs = ["."] + list(w) + ["."]
        # well this is just rust window of 2: emma -> em, mm, ma
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1
    # this is random generator with seed so outcomes are deterministic
    generator = torch.Generator().manual_seed(2147483647)
    # we do we add 1? To do model smoothing, we just do want to get prob of 0 because that would make log function to return -inf
    # we just add one so unlikely names like 'adrejq' are still possible but rather of small chances
    P = (N + 1).float()
    # we want to have sum across rows thus 1(0 means sum of columns)
    # this normalizes counts, it's now probability so chars that occur often will be close to 1 and others to 0
    # /= may be faster than x = x /... because memory allocation
    P /= P.sum(1, keepdim=True)
    for _ in range(10):
        out = []
        ix = 0
        while True:
            # probability for char related with ix (1 - 'a', 2 - 'b' etc)
            p = P[ix]
            # this is fancy math lib that takes probabilities and returns vector of indexes based on it.
            # It other words if we have p = [0.5, 0.1, 0.4] and num_samples = 10 then index 0 should occur 5 times, 1 single time and 2 four times
            # replacement = True allows to reuse drawn values otherwise there are taken out of "next lottery"
            ix = torch.multinomial(
                p, num_samples=1, replacement=True, generator=generator
            ).item()
            out.append(itos[ix])
            # 0 means end of sequence char
            if ix == 0:
                break
        print("".join(out))
    measure_learning_efficiency(words, P, stoi)


if __name__ == "__main__":
    words = open("names.txt").read().splitlines()
    shortest = min(len(w) for w in words)
    biagram_statistic_model(words)
