import argparse
import csv
import numpy as np

_data = []
_attributes = []
_root = []


def main():
    args = parser.parse_args()
    data = args.data

    with open(data) as csvFile:
        _file = csv.reader(csvFile, delimiter=",")  # Separate the columns at ','
        for n in _file:
            _data.append(n)  # add to data list to process

    highestLength = 0
    for item in _data:
        if len(item) > highestLength:
            highestLength = len(item)  # so we can get the max length and ignore items with length 0

    for item in _data:  # remove the data points belonging to such a case^
        if len(item) < highestLength:
            _data.remove(item)

    for i in range(len(_data[0])):
        _attributes.append([])

    for instanceIndex in range(len(_data)):
        for attIndex in range(len(_data[instanceIndex])):
            if not contains(_attributes[attIndex], _data[instanceIndex][attIndex]):
                _attributes[attIndex].append(_data[instanceIndex][attIndex])

    for i in range(len(_data[0]) - 1):  # building root nodes with \n
        _root.append("\n")

    buildTree(_root, 0, "root")


def entropy(attributes, withTotalNumber=False):
    _class = len(_attributes[-1])

    no_of_class_instances = []  # Count number of instances belonging to a class
    for x in range(_class):
        no_of_class_instances.append(0)

    for x in _data:  # Count number of instances that suit to our class
        suits_atts = True
        for attValInd in range(len(x) - 1):
            if attributes[attValInd] == "\n":
                continue
            elif attributes[attValInd] != x[attValInd]:
                suits_atts = False
        if suits_atts:
            no_of_class_instances[_attributes[-1].index(x[-1])] += 1

    total = 0
    for x in no_of_class_instances:
        total += x

    entropy_value = 0

    for x in no_of_class_instances:
        if x != 0:
            entropy_value += x / total * (
                    np.log(x / total) / np.log(_class))

    entropy_value *= -1  # entropy value has a negative before the summation

    if withTotalNumber:
        return [entropy_value, total]
    else:
        return entropy_value


def gain(attributes, nextAttInd):
    entropyS = entropy(attributes, True)  # Total entropy

    entropies_of_att_val_with_number = []

    for att_val in _attributes[nextAttInd]:
        new_node = attributes.copy()
        new_node[nextAttInd] = att_val
        entropies_of_att_val_with_number.append(entropy(new_node, True))

    _gain = entropyS[0]
    for entropy_i in entropies_of_att_val_with_number:
        _gain -= (entropy_i[1] / entropyS[1]) * entropy_i[0]  # Standard Gain Formula

    return _gain


def NextAtt(attributes):
    highest_gain = [gain(attributes, 0), 0]  # Gain,Index
    for x in range(1, len(_attributes) - 1):  # To not include class variable
        current_gain = gain(attributes, x)
        if current_gain > highest_gain[0]:
            highest_gain = [current_gain, x]

    return highest_gain[1]  # return the highest gain value


def which_class(attributes):
    distro = []
    for x in range(len(_attributes[-1])):
        distro.append(0)

    for x in _data:
        suits_atts = True
        for att_val_ind in range(len(x) - 1):
            if attributes[att_val_ind] == "\n":
                continue  # skip to the next
            elif attributes[att_val_ind] != x[att_val_ind]:  # if they are not equal, it does not suit our Attribute
                suits_atts = False
        if suits_atts:
            distro[_attributes[-1].index(x[-1])] += 1  # Counter to note the class index

    return np.argmax(distro)


def contains(array, newItem):  # simple function to check if an item exists in our list
    for x in array:
        if x == newItem:
            return True
    return False


def buildTree(attributes, depth, attInNV):
    data_point = [depth, attInNV, entropy(attributes), "no_leaf"]

    if (entropy(attributes) < 0.0000001) or depth >= len(
            _attributes) - 1:  # Entropy is very low or we have reached the maximum depth
        data_point[-1] = _attributes[-1][which_class(attributes)]
        print(data_point[0], ",", data_point[1], ",", data_point[2], ",", data_point[3])
    else:
        print(data_point[0], ",", data_point[1], ",", data_point[2], ",", data_point[3])

        aI = NextAtt(attributes)

        splits = []

        for att_val in _attributes[aI]:
            temp = attributes.copy()
            temp[aI] = att_val
            splits.append(temp)

        for split_ind in range(len(splits)):
            buildTree(splits[len(splits) - 1 - split_ind], depth + 1,
                      "att{}={}".format(aI, splits[len(splits) - 1 - split_ind][aI]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")  # read data from the terminal
    main()
