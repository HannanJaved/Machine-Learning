import argparse
import csv
import numpy


def toArray(data):
    reader = csv.reader(args.data)
    for row in reader:
        data.append(row)

    for row in range(len(data)):
        for item in range(1, len(data[row])):
            data[row][item] = float(data[row][item])

    return data


def distance(x, y):
    _sum = 0
    for i in range(1, len(x)):
        _sum += numpy.linalg.norm(x[i] - y[i]) ** 2
    _sum = _sum ** (1 / 2)
    return _sum


def classifier(c, kN):
    classes = [0, 0]  # Count number of instances for each Class A and B
    dk = distance(c, kN[0])
    d_one = distance(c, kN[len(kN) - 1])

    for n in kN:
        di = distance(c, n)
        w = (dk - di) / (dk - d_one)
        if n[0] == 'A':
            classes[0] += w  # Increase count at index 0 for Class A
        elif n[0] == 'B':
            classes[1] += w  # Increase count at index 1 for Class B

    return 'A' if classes[0] > classes[1] else 'B'


def insertK(a, item, c, _k):
    _insert = False
    dist = distance(c, item)
    Length = len(a)

    for i in range(Length):
        if dist >= distance(c, a[i]):
            a.insert(i, item)
            _insert = True
            break

    if not _insert:
        a.append(item)

    if len(a) > k:
        a.remove(a[0])

    return a


def IB2(d):
    CaseBase = [d[0]]
    d.pop(0)

    i = 0
    while i < len(d):
        c = d[i]

        nC = CaseBase[0]

        for CaseBaseC in CaseBase:
            if distance(c, CaseBaseC) < distance(c, nC):
                nC = CaseBaseC

        if c[0] != nC[0]:
            CaseBase.append(c)
            d.pop(i)
        i += 1

    return d, CaseBase


def _printOutput(CaseBase):
    for i in CaseBase:
        s = ""

        for step, entry in enumerate(i):
            s += str(entry)
            if step != len(i) - 1:
                s += ","
        print(s)


def main():
    data = []
    data = toArray(data)
    data, cb = IB2(data)

    total = 0
    for p in data:
        kN = []
        for c in cb:
            if len(kN) < k:
                kN = insertK(kN, c, p, k)
            elif distance(p, c) < distance(p, kN[0]):
                kN = insertK(kN, c, p, k)

        clas = classifier(p, kN)
        if clas != p[0]:
            total += 1

    print(total)
    _printOutput(cb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=argparse.FileType())
    parser.add_argument('--k', type=int)
    args = parser.parse_args()
    k = args.k
    main()
