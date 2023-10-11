import argparse
import csv
import numpy as np


def calculateGradient(w, x, y, fx, eta):  # x - data point, y - target output,eta - learning rate,fx - linear function
    gradient = x * (y - fx)
    gradient = np.sum(gradient, axis=0)
    eta_gradient = np.array(eta * gradient).reshape(w.shape)
    w = w + eta_gradient
    return gradient, w


def main():
    args = parser.parse_args()
    data, eta, threshold = args.data, float(args.eta), float(args.threshold)

    with open(data) as csvFile:
        file = csv.reader(csvFile, delimiter=',')
        x1 = []
        x2 = []

        for n in file:
            x1.append([1.0] + n[:-1])
            x2.append([n[-1]])

    x1 = np.array(x1).astype(float)
    x2 = np.array(x2).astype(float)
    w = np.zeros(x1.shape[1]).astype(float)
    w = w.reshape(x1.shape[1], 1)

    f = np.dot(x1, w)
    prev_sse = np.sum(np.square(f - x2))

    print(*[0], *["{0:}".format(value) for value in w.T[0]], *["{0:}".format(prev_sse)])

    gradient, w = calculateGradient(w, x1, x2, f, eta)

    iteration = 1
    while True:
        f = np.dot(x1, w)
        sse = np.sum(np.square(f - x2))

        if abs(sse - prev_sse) > threshold:
            print(
                *[iteration], *["{0:}".format(val) for val in w.T[0]], *["{0:}".format(sse)])
            gradient, w = calculateGradient(w, x1, x2, f, eta)
            prev_sse = sse
            iteration += 1
        else:
            break

    print(*[iteration], *["{0:}".format(val) for val in w.T[0]], *["{0:}".format(sse)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--eta")
    parser.add_argument("--threshold")
    main()
