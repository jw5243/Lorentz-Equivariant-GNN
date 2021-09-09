from random import random
from math import cos, sin, pi, sqrt
from copy import deepcopy
import torch
from lorentz_equivariant_gnn.legnn_model import LEGNN
from data_loader import get_edges
from statistics import stdev, mean
from matplotlib import pyplot as plt


def rotate_x(lorentz_vector, theta):
    c, s = cos(theta), sin(theta)
    rotation_matrix = torch.tensor([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, c, -s],
                                    [0, 0, s, c]])
    return (rotation_matrix @ lorentz_vector.transpose(0, 1)).transpose(0, 1)


def rotate_y(lorentz_vector, theta):
    c, s = cos(theta), sin(theta)
    rotation_matrix = torch.tensor([[1, 0, 0, 0],
                                    [0, c, 0, s],
                                    [0, 0, 1, 0],
                                    [0, -s, 0, c]])
    return (rotation_matrix @ lorentz_vector.transpose(0, 1)).transpose(0, 1)


def rotate_z(lorentz_vector, theta):
    c, s = cos(theta), sin(theta)
    rotation_matrix = torch.tensor([[1, 0, 0, 0],
                                    [0, c, -s, 0],
                                    [0, s, c, 0],
                                    [0, 0, 0, 1]])
    return (rotation_matrix @ lorentz_vector.transpose(0, 1)).transpose(0, 1)


def boost_x(lorentz_vector, gamma):
    beta = sqrt(1 - 1 / (gamma ** 2))
    boost_matrix = torch.tensor([[gamma, -beta * gamma, 0, 0],
                                 [-beta * gamma, gamma, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
    return (boost_matrix @ lorentz_vector.transpose(0, 1)).transpose(0, 1)


def boost_y(lorentz_vector, gamma):
    beta = sqrt(1 - 1 / (gamma ** 2))
    boost_matrix = torch.tensor([[gamma, 0, -beta * gamma, 0],
                                 [0, 1, 0, 0],
                                 [-beta * gamma, 0, gamma, 0],
                                 [0, 0, 0, 1]])
    return (boost_matrix @ lorentz_vector.transpose(0, 1)).transpose(0, 1)


def boost_z(lorentz_vector, gamma):
    beta = sqrt(1 - 1 / (gamma ** 2))
    boost_matrix = torch.tensor([[gamma, 0, 0, -beta * gamma],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [-beta * gamma, 0, 0, gamma]])
    return (boost_matrix @ lorentz_vector.transpose(0, 1)).transpose(0, 1)


def rotate(lorentz_vector, theta_x, theta_y, theta_z):
    # Rotate in the order of x, y, and z
    return rotate_z(rotate_y(rotate_x(lorentz_vector, theta_x), theta_y), theta_z)


def boost(lorentz_vector, gamma_x, gamma_y, gamma_z):
    # Rotate in the order of x, y, and z
    return boost_z(boost_y(boost_x(lorentz_vector, gamma_x), gamma_y), gamma_z)


def finalize_network_output(output):
    output = torch.mean(output, dim = 1).squeeze(1)
    return torch.sigmoid(output)


def test_equivariance(network, edges, feature_vector, lorentz_vector, boost, debug = False):
    boosted_input = boost_x(lorentz_vector, boost)
    h = feature_vector.unsqueeze(0)
    x = lorentz_vector.unsqueeze(0)
    output1, x = network(h, x, edges)
    x = x.squeeze(0)
    post_boost_output = boost_x(x, boost)

    boosted_input = boosted_input.unsqueeze(0)
    output2, boosted_output = network(h, boosted_input, edges)

    output1 = finalize_network_output(output1)
    output2 = finalize_network_output(output2)

    if debug:
        #print(torch.isclose(post_boost_output, boosted_output, atol = 1e-8, rtol = 1e-3))
        print(output1[0].item() - output2[0].item())

    return abs(output1[0].item() - output2[0].item())


if __name__ == '__main__':
    #test_boost_x = 2 * random() + 1
    test_boost_x = 10000

    #print(test_boost_x)

    # Dummy parameters
    n_nodes = 4
    n_feat = 1
    x_dim = 4

    # Dummy variables h, x and fully connected edges
    h = torch.rand(n_nodes, n_feat)
    x = torch.rand(n_nodes, x_dim)
    edges = get_edges(n_nodes)

    #print("Input: " + str(x))

    count = 200
    networks = []

    # Initialize LEGNN
    for i in range(count):
        legnn = LEGNN(input_feature_dim = n_feat, message_dim = 16, output_feature_dim = 1, edge_feature_dim = 0,
                      n_layers = 3)
        networks.append(legnn)

    mean_relative_deviation = []
    error_bars = []
    boost_list = []
    for i in range(189):
        error_list = []
        relative_deviation = []
        test_boost_x = (1 + 0.05) ** i
        boost_list.append(test_boost_x)
        for j in range(count):
            h = torch.rand(n_nodes, n_feat)
            x = torch.rand(n_nodes, x_dim)
            error = test_equivariance(LEGNN(input_feature_dim = n_feat, message_dim = 16, output_feature_dim = 1, edge_feature_dim = 0,
                      n_layers = 3), edges, h, x, test_boost_x)
            error_list.append(error)
            relative_deviation.append(error)

        mean_relative_deviation.append(mean(relative_deviation))
        error_bars.append(stdev(error_list))
        print(mean_relative_deviation[-1])
        print(error_bars[-1])

    axes = plt.axes()
    axes.set_xscale('log')
    axes.set_yscale('log')
    plt.plot(boost_list, mean_relative_deviation)
    plt.xlabel("Lorentz Boost Factor (γ)")
    plt.ylabel("Relative Deviation in Network Output")
    #plt.errorbar(boost_list, mean_relative_deviation, error_bars)
    plt.show()
