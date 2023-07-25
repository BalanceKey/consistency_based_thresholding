import zipfile
import numpy as np
import matplotlib.pyplot as plt

def network_density(conn):
    ''' Assuming the connectivity matrix is undirected
        Returns the network density aka the proportion of actual
        connections in the network relative to all possible connections '''
    N = conn.shape[0]
    assert np.array_equal(conn, conn.transpose()) # assuming matrix is symmetrical
    upper_triangular = np.triu(conn)
    existing_edges = np.argwhere(upper_triangular > 0)
    N_actual_connections = existing_edges.shape[0]
    N_possible_connections = N*(N-1)/2
    return N_actual_connections/N_possible_connections


def read_SC_matrices(retro_data_path, patients_list):
    N = 30
    patients_list = np.loadtxt(patients_list, dtype=str)
    SC_list = np.empty(shape=(162, 162, 30))
    tract_lengths_list = np.empty(shape=(162, 162, 30))
    for i in range(N):
        try:
            with zipfile.ZipFile(
                    f'{retro_data_path}/{patients_list[i]}/tvb/connectivity.vep.zip') as sczip:
                with sczip.open('weights.txt') as weights:
                    SC = np.loadtxt(weights)
                with sczip.open('tract_lengths.txt') as lengths:
                    tract_lengths = np.loadtxt(lengths)
        except FileNotFoundError as err:
            print(f'{err}: Structural connectivity not found for {patients_list[i]}')

        # ignore self-connections
        SC[np.diag_indices(SC.shape[0])] = 0
        # normalize each SC
        SC = (SC - SC.min()) / (SC.max() - SC.min())
        SC_list[:, :, i] = SC
        tract_lengths_list[:, :, i] = tract_lengths
    return SC_list, tract_lengths_list

def threshold_consistency(Ws, p):
    '''
    This function thresholds the group mean connectivity matrix of a set of
    connectivity matrices by preserving a proportion of p (0<p<1) of the
    edges with the smallest coefficient of variation accross the group. All
    other weigths and all the weights on the main diagonal (self-self connections)
    are set to 0.

    :param Ws: NxNxM group of M weighted connectivity matrices
    :param p: proportion of weights to preserve in [0, 1] interval
    :return: Wmean, thresholded group mean connectivity matrix

    Reference: Roberts and Breakspear (2016)
    '''
    Wmean = np.average(Ws, axis=2) #+ 0.0000000000001  # group mean connectivity matrix
    Wstd = np.std(Ws, axis=2)       # group standard deviation matrix
    with np.errstate(divide='ignore', invalid='ignore'):
        Wcv = np.true_divide(Wstd, Wmean)  # coefficient of variation (CV) across the group
        Wcv = np.nan_to_num(Wcv)           # replace nan values with 0 (due to division by 0)

    # plt.figure()
    # plt.scatter(np.log(Wmean.flatten()), np.log(Wcv.flatten()))
    # plt.xlabel('Log Weights')
    # plt.ylabel('Log CV')
    # plt.show()

    N = Wmean.shape[0]  # number of nodes
    if np.array_equal(Wmean, Wmean.transpose()):    # if symmetric matrix
        Wmean = np.triu(Wmean)                      # ensure symmetry is preserved
        ud = 2                                      # halve number of removed links
    else:
        ud = 1

    ind = np.argwhere(Wmean>0)                      # find all links
    M = Wcv
    ind_CV = np.empty(shape=(ind.shape[0], 1))
    for i in range(ind.shape[0]):
        ind_CV[i] = M[ind[i][0], ind[i][1]]
    print(np.argwhere(np.isnan(ind_CV)))
    ind_M = np.append(ind, ind_CV, axis=1)          # sort by CV (keep the lowest CV, remove the strongest CV)
    E = ind_M[ind_M[:, 2].argsort()]                # sort from lowest to strongest CV (third column)
    en = round((N**2-N)*p/ud)                       # number of links to be preserved
    E_to_remove =  E[en + 1:, :-1].astype('int')    # apply threshold, keep lowest CVs == high consistency
    for i in E_to_remove:
        Wmean[i[0], i[1]] = 0

    if ud == 2:                                      # if symmetric matrix
        Wmean = Wmean + Wmean.T                      # reconstruct symmetry
    return Wmean

def main():
    retro_data_path = '/Users/dollomab/MyProjects/Epinov_trial/retrospective_patients'
    patients_list = f'{retro_data_path}/sublist.txt'
    SC_list, tract_lengths_list = read_SC_matrices(retro_data_path, patients_list)

    plt.figure()
    plt.imshow(np.log(SC_list[:,:,18]))
    plt.show()

    network_densities = []
    for i in range(SC_list.shape[2]):
        network_densities.append(network_density(SC_list[:, :, i]))

    # Standard method: thresholding the connection weight
    Wmean_std = np.average(SC_list, axis=2)
    Wline, Wcol = np.where(Wmean_std < 0.00002)
    for i in range(Wline.size):
        Wmean_std[Wline[i], Wcol[i]] = 0
    plt.figure()
    plt.imshow(np.log(Wmean_std))
    plt.show()

    # Consistency-based thresholding method
    Ws = SC_list
    p = np.mean(network_densities)
    Wmean_thr = threshold_consistency(Ws, p)

    plt.figure()
    plt.imshow(np.log(Wmean_thr))
    plt.show()

    # some histograms just for fun to compare the two methods, like in the Roberts et al. 2017 paper
    plt.figure()
    plt.hist(np.log(np.average(SC_list, axis=2).flatten() + 0.000000001), bins=70, histtype='step', facecolor='blue', label='AVG')
    plt.hist(np.log(Wmean_std.flatten() + 0.000000001), bins=70, histtype='step', facecolor='black', label='STAND')
    plt.hist(np.log(Wmean_thr.flatten() + 0.000000001), bins=70, histtype='step', facecolor='red', label='CONSIST')
    plt.xlabel('Log(weight)')
    plt.ylabel('Counts')
    plt.legend()
    plt.show()
