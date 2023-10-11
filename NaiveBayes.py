import argparse
import csv
import numpy as np

data = []
splitData = [[], []]
pci = [[0, 0], [0, 0]]


def _init():
    reader = csv.reader(args.data)

    for row in reader:
        data.append(row)
    for item in data:
        splitData[_class(item[0])].append(item)


def classifier(i, mu, s, pc):

    for ClassInstance in range(2):
        for a in range(2):
            f1 = (1 / (2 * np.pi * s[ClassInstance][a]) ** (1 / 2))
            f2 = float(i[a + 1]) - mu[ClassInstance][a]
            f2 = -(f2 ** 2) / (2 * s[ClassInstance][a])
            f2 = np.exp(f2)

            pci[ClassInstance][a] = f1 * f2

    return 0 if (pc[0] * pci[0][0] * pci[0][1] >= pc[1] * pci[1][0] * pci[1][1]) else 1


def _class(c):
    return (1, 0)[c == 'A']


def _print(mu, s, pc):
    print(mu[0][0], s[0][0], mu[0][1], s[0][1], pc[0], sep=',')
    print(mu[1][0], s[1][0], mu[1][1], s[1][1], pc[1], sep=',')

    total = 0

    for i in data:
        c = classifier(i, mu, s, pc)
        if c != _class(i[0]):
            total += 1

    print(total)


def main():
    nc = [len(splitData[0]), len(splitData[1])]
    pc = [nc[0] / (nc[0] + nc[1]),
          nc[1] / (nc[0] + nc[1])]

    mu = [[0, 0], [0, 0]]
    for i in data:
        mu[_class(i[0])][0] += float(i[1])
        mu[_class(i[0])][1] += float(i[2])

    for ci in range(len(mu)):
        mu[ci][0] *= 1 / nc[ci]
        mu[ci][1] *= 1 / nc[ci]

    sigma = [[0, 0], [0, 0]]
    for i in data:
        sigma[_class(i[0])][0] += (float(i[1]) - mu[_class(i[0])][0]) ** 2
        sigma[_class(i[0])][1] += (float(i[2]) - mu[_class(i[0])][1]) ** 2

    for ci in range(len(mu)):
        sigma[ci][0] *= 1 / (nc[ci] - 1)
        sigma[ci][1] *= 1 / (nc[ci] - 1)

    _print(mu, sigma, pc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=argparse.FileType('r'))
    args = parser.parse_args()
    _init()
    main()
