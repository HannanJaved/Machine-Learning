import argparse
import numpy as np

w_bias_ = [0.2, -0.5, 0.3, -0.1]
w_a_h = [-0.3, -0.1, 0.2]
w_b_h = [0.4, -0.4, 0.1]
w_h_o = [0.1, 0.3, -0.4]


def _arguments():
    args = parser.parse_args()
    fp, lr, i = args.data, args.eta, args.iterations
    data = np.genfromtxt(fp, delimiter=',')
    return data, lr, i


def sigmoidActivation(net_h):
    sigmoidActivation_h = 1 / (1 + np.exp(-net_h))
    return sigmoidActivation_h


def _netH(wah, ia, wbh, ib, wbias):
    net_h = wah * ia + wbh * ib + wbias * 1
    return net_h


def SNN(d, lR, itr):
    eta = lR

    print('a', 'b', 'h1', 'h2', 'h3', 'o', 't', 'delta_h1', 'delta_h2', 'delta_h3', 'delta_o', 'w_bias_h1', 'w_a_h1',
          'w_b_h1', 'w_bias_h2', 'w_a_h2', 'w_b_h2', 'w_bias_h3', 'w_a_h3', 'w_b_h3', 'w_bias_o', 'w_h1_o', 'w_h2_o',
          'w_h3_o', sep=',')

    print('-', '-', '-', '-', '-', '-', '-', '-', '-', '-', '-', w_bias_[0], w_a_h[0], w_b_h[0], w_bias_[1], w_a_h[1],
          w_b_h[1],
          w_bias_[2], w_a_h[2], w_b_h[2], w_bias_[3], w_h_o[0], w_h_o[1], w_h_o[2], sep=',')

    i = 0
    while i < itr:

        for row in d:

            Input_a = row[0]
            Input_b = row[1]

            net_h1 = _netH(w_a_h[0], Input_a, w_b_h[0], Input_b, w_bias_[0])
            net_h2 = _netH(w_a_h[1], Input_a, w_b_h[1], Input_b, w_bias_[1])
            net_h3 = _netH(w_a_h[2], Input_a, w_b_h[2], Input_b, w_bias_[2])

            sA_h1 = sigmoidActivation(net_h1)
            sA_h2 = sigmoidActivation(net_h2)
            sA_h3 = sigmoidActivation(net_h3)

            net_output = sA_h1 * w_h_o[0] + sA_h2 * w_h_o[1] + sA_h3 * \
                         w_h_o[2] + 1 * w_bias_[3]
            sigmoidActivation_output = sigmoidActivation(net_output)

            error = row[2] - sigmoidActivation_output

            if error:
                delta_o = sigmoidActivation_output * (1 - sigmoidActivation_output) * (
                        row[2] - sigmoidActivation_output)
                delta_h1 = sA_h1 * (1 - sA_h1) * (w_h_o[0] * delta_o)
                delta_h2 = sA_h2 * (1 - sA_h2) * (w_h_o[1] * delta_o)
                delta_h3 = sA_h3 * (1 - sA_h3) * (w_h_o[2] * delta_o)

                w_bias_[0] += (eta * delta_h1 * 1)
                w_a_h[0] += (eta * delta_h1 * Input_a)
                w_b_h[0] += (eta * delta_h1 * Input_b)
                w_bias_[1] += (eta * delta_h2 * 1)
                w_a_h[1] += (eta * delta_h2 * Input_a)
                w_b_h[1] += (eta * delta_h2 * Input_b)
                w_bias_[2] += (eta * delta_h3 * 1)
                w_a_h[2] += (eta * delta_h3 * Input_a)
                w_b_h[2] += (eta * delta_h3 * Input_b)

                w_bias_[3] += (eta * delta_o * 1)
                w_h_o[0] += (eta * delta_o * sA_h1)
                w_h_o[1] += (eta * delta_o * sA_h2)
                w_h_o[2] += (eta * delta_o * sA_h3)

                arr = np.array((row[0], row[1], sA_h1, sA_h2, sA_h3,
                                sigmoidActivation_output, row[2], delta_h1, delta_h2, delta_h3, delta_o, w_bias_[0],
                                w_a_h[0], w_b_h[0], w_bias_[1], w_a_h[1], w_b_h[1], w_bias_[2], w_a_h[2], w_b_h[2],
                                w_bias_[3], w_h_o[0],
                                w_h_o[1], w_h_o[2]))

                print(*list(np.round(arr, 5)), sep=',')

        i += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--eta", type=float)
    parser.add_argument("--iterations", type=int)
    file, learning_rate, iterations = _arguments()
    SNN(file, learning_rate, iterations)
