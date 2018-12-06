import argparse
import numpy as np 


def converge_state(old_state, edges, averages):
    new_state = np.zeros(len(old_state))
    
    for i in range(len(old_state)):
        current_value = old_state[i]

        if current_value < edges[i][0]:
            new_state[i] = averages[i][0]

        elif current_value < edges[i][1]:
            new_state[i] = averages[i][1]

        elif current_value < edges[i][2]:
            new_state[i] = averages[i][2]

        elif current_value < edges[i][3]:
            new_state[i] = averages[i][3]

        elif current_value < edges[i][4]:
            new_state[i] = averages[i][4]

        elif current_value < edges[i][5]:
            new_state[i] = averages[i][5]

        elif current_value < edges[i][6]:
            new_state[i] = averages[i][6]

        elif current_value < edges[i][7]:
            new_state[i] = averages[i][7]

        elif current_value < edges[i][8]:
            new_state[i] = averages[i][8]

        elif current_value < edges[i][9]:
            new_state[i] = averages[i][9]

        elif current_value < edges[i][10]:
            new_state[i] = averages[i][10]
        else:
            new_state[i] = averages[i][11]
        # print('New value: ', new_state[i])

    return new_state   


def compute_bins(c_pos_bounds, c_vel_bounds, p_pos_bounds, p_vel_bounds, n_bins=10):

    c_pos_edges = np.linspace(c_pos_bounds[0], c_pos_bounds[1], n_bins+1)
    c_pos_values = [(c_pos_edges[i] + c_pos_edges[i+1])/2 for i in range(len(c_pos_edges)-1)]
    c_pos_values.append(c_pos_bounds[1])
    c_pos_values.insert(0, c_pos_bounds[0])

    c_vel_edges = np.linspace(c_vel_bounds[0], c_vel_bounds[1], n_bins+1)
    c_vel_values = [(c_vel_edges[i] + c_vel_edges[i+1])/2 for i in range(len(c_vel_edges)-1)]
    c_vel_values.append(c_vel_bounds[1])
    c_vel_values.insert(0, c_vel_bounds[0])

    p_pos_edges = np.linspace(p_pos_bounds[0], p_pos_bounds[1], n_bins+1)
    p_pos_values = [(p_pos_edges[i] + p_pos_edges[i+1])/2 for i in range(len(p_pos_edges)-1)]
    p_pos_values.append(p_pos_bounds[1])
    p_pos_values.insert(0, p_pos_bounds[0])

    p_vel_edges = np.linspace(p_vel_bounds[0], p_vel_bounds[1], n_bins+1)
    p_vel_values = [(p_vel_edges[i] + p_vel_edges[i+1])/2 for i in range(len(p_vel_edges)-1)]
    p_vel_values.append(p_vel_bounds[1])
    p_vel_values.insert(0, p_vel_bounds[0])

    all_edges = [c_pos_edges, c_vel_edges, p_pos_edges, p_vel_edges]
    all_values = [c_pos_values, c_vel_values, p_pos_values, p_vel_values]

    return all_edges, all_values


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--c_pos_bounds', type = list, default = [-2, 2],
                        help='List with the minumum and maximum value for the position of the cart')

    parser.add_argument('--c_vel_bounds', type = list, default = [-3, 3],
                        help='List with the minumum and maximum value for the velocity of the cart')

    parser.add_argument('--p_pos_bounds', type = list, default = [-0.21, 0.21],
                        help='List with the minumum and maximum value for the position of the pole')
    
    parser.add_argument('--p_vel_bounds', type = list, default = [-3, 3],
                        help='List with the minumum and maximum value for the position of the cart')

    parser.add_argument('--n_bins', type = int, default = 10,
                        help='The number of bins to discretize the continous state space')

    args, unparsed = parser.parse_known_args()

    edges, averages = compute_bins()

    test_state = np.array([0.03432708,  0.0007369 , 0.5, -0.04862195])
    print('Old state: ', test_state)
    new_state = converge_state(test_state, edges, averages)
    print('New state: ', new_state)