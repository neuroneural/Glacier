import time
from collections import deque
from itertools import chain
import numpy as np
import torch
import sys
import os
# sys.path.append(os.path.abspath("/data/users2/umahmood1/Glacier/src"))
from scipy import stats
import torch.nn as nn

from src.utils import get_argparser
from src.encoders_ICA import NatureCNN

import pandas as pd
import datetime
from src.All_Architecture_without_lstm import combinedModel
# a= combinedModel(1,2,3)
from tqdm import tqdm
from src.graph_the_works_fMRI_glacier import the_works_trainer


def find_indices_of_each_class(all_labels):
    HC_index = (all_labels == 0).nonzero()
    SZ_index = (all_labels == 1).nonzero()

    return HC_index, SZ_index


def train_encoder(args):
    start_time = time.time()




    print("testing")





    # ID = args.script_ID + 3
    ID = args.script_ID - 1
    JobID = args.job_ID


    ID = 0
    print('ID = ' + str(ID))
    print('exp = ' + args.exp)
    print('pretraining = ' + args.pre_training)
    sID = str(ID)
    currentDT = datetime.datetime.now()
    d1 = currentDT.strftime("%Y-%m-%d%H:%M:%S")
    d2 = '_' + str(JobID) + '_ startFold_' + str(args.start_CV) + '_' + str(args.cv_Set)

    Name = args.exp + '_FBIRN_' + args.pre_training + 'Glacier_HardSwish-defaul-order'
    dir = 'run-' + d1 + d2 + Name
    dir = dir + '-' + str(ID)
    wdb = 'wandb_new'

    wpath = os.path.join(os.getcwd(), wdb)
    path = os.path.join(wpath, dir)
    args.path = path
    os.mkdir(path)

    wdb1 = 'wandb_new'
    wpath1 = os.path.join(os.getcwd(), wdb1)


    p = 'UF'
    dir = 'run-2019-09-1223:36:31' + '-' + str(ID) + 'FPT_ICA_COBRE'
    p_path = os.path.join(os.getcwd(), p)
    p_path = os.path.join(p_path, dir) 
    args.p_path = p_path
    # os.mkdir(fig_path)
    # hf = h5py.File('../FBIRN_AllData.h5', 'w')
    tfilename = str(JobID) + 'outputFILENEWONE' + Name + str(ID)

    output_path = os.path.join(os.getcwd(), 'Output')
    output_path = os.path.join(output_path, tfilename)
    # output_text_file = open(output_path, "w+")
    # writer = SummaryWriter('exp-1')
    ntrials = args.ntrials
    ngtrials = 10
    best_auc = 0.

    tr_sub_SZ = [142] #142, 132 80
    tr_sub_HC = [134] #134, 124 74



   
    gain = [1, 1, 1, 1, 2.25, 1]  # NPT

    sub_per_class_SZ = tr_sub_SZ[ID]
    sub_per_class_HC = tr_sub_HC[ID]

    sample_x = 100
    sample_y = 1
    subjects = 311
    tc = 160

    samples_per_subject = int(tc / sample_y)
    samples_per_subject = 160
    ntest_samples_perclass_SZ = 9
    ntest_samples_perclass_HC = 8
    nval_samples_perclass_SZ = 9
    nval_samples_perclass_HC = 8
    test_start_index = 0
    test_end_index = test_start_index + ntest_samples_perclass_SZ
    window_shift = 1

    if torch.cuda.is_available():
        cudaID = str(torch.cuda.current_device())
        print(torch.cuda.device_count())
        device = torch.device("cuda:0")
        device2 = torch.device("cuda:0")
        # device = torch.device("cuda:" + str(args.cuda_id))
    else:
        device = torch.device("cpu")
    print('device = ', device)
    print('device = ', device2)



    n_good_comp = 53
    n_regions = 100



    with open('../DataandLabels/FBIRN_alldata_new_160.npz', 'rb') as file: # input data, should be of shape (n_subjects, n_components, n_time_points) e.g. (311, 100, 160)
        data = np.load(file)

    print(data.shape)
    n_subjects = data.shape[0]
    n_regions = data.shape[1]
    tc = data.shape[2]
    data[data != data] = 0


    for t in range(subjects):
        for r in range(n_regions):
            data[t, r, :] = stats.zscore(data[t, r, :])

    # data = data + 2
    data = torch.from_numpy(data).float()
    finalData = np.zeros((subjects, samples_per_subject, n_regions, sample_y))
    for i in range(subjects):
        for j in range(samples_per_subject):
            #if j != samples_per_subject-1:
            finalData[i, j, :, :] = data[i, :, (j * window_shift):(j * window_shift) + sample_y]
            #else:
                #finalData[i, j, :, :17] = data[i, :, (j * window_shift):]


    finalData2 = torch.from_numpy(finalData).float()
    # selected = np.arange(n_subjects) != 73
    # finalData2 = finalData2[selected,:,:,:]#torch.cat((finalData2[0:73,:,:,:], finalData2[74:,:,:,:]), dim=0)
    finalData2[finalData2 != finalData2] = 0



    filename = '../DataandLabels/index_array_labelled_FBIRN_temp.csv'
    df = pd.read_csv(filename, header=None)
    index_array = df.values
    index_array = torch.from_numpy(index_array).long()
    index_array = index_array.view(subjects-1)

    filename = '../DataandLabels/labels_FBIRN_new.csv'
    df = pd.read_csv(filename, header=None)
    all_labels = df.values
    all_labels = torch.from_numpy(all_labels).int()
    all_labels = all_labels.view(subjects)
    all_labels = all_labels - 1
    # all_labels = all_labels[selected]

    finalData2 = finalData2[:, 0:155, :,:] #index_array at first index can be used to permute the order of subjects and labels
    all_labels = all_labels[:]
    tc = 155
    print(finalData2.shape)
    # return

    # finalData2_copy = torch.clone(finalData2)

    # finalData2_copy = torch.squeeze(finalData2_copy)
    # finalData2_copy = finalData2_copy.permute(0,2,1)
    # cor, rho = stats.spearmanr(finalData2[0,0,:,:],axis=1)
    # print(cor.shape)
    # return

    # FNC = np.zeros((subjects - 1, 4950))  # 6670
    # corrM = np.zeros((subjects-1, n_regions, n_regions))
    # for i in range(subjects - 1):
    #     corrM[i, :, :] = np.corrcoef(finalData2_copy[i])
    #     M = corrM[i, :, :]
    #     FNC[i, :] = M[np.triu_indices(n_regions, k=1)]
    # corrM = torch.from_numpy(corrM).float()
    # print (corrM[0,0,:])
    # print(FNC.shape)

    # report = poly(FNC, all_labels, n_folds=19, exclude=['RBF SVM'])
    # # Plot results
    # report.plot_scores()
    # return

    # all_labels = torch.randint(high=2, size=(311,), dtype=torch.int64)
    test_indices_HC = [0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136]
    test_indices_SZ = [0, 9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108, 117, 126, 135, 144, 153]

    

    number_of_cv_sets = args.cv_Set
    n_regions_output = n_regions
    tc_after_encoder = 155
    HC_index, SZ_index = find_indices_of_each_class(all_labels)
    print(HC_index.shape)
    print(SZ_index.shape)
    # return
    total_test_size = ntest_samples_perclass_HC + ntest_samples_perclass_SZ
    results = torch.zeros(ntrials * number_of_cv_sets, 10)
    # adjacency_matrices_FNC = torch.zeros(ntrials * number_of_cv_sets, total_test_size, n_regions_output,
    #                                          n_regions_output)
    adjacency_matrices_learned = torch.zeros(ntrials * number_of_cv_sets, total_test_size, n_regions_output,
                                             n_regions_output)

    # temporal_adjacency_matrices_learned = torch.zeros(ntrials * number_of_cv_sets, total_test_size, n_regions_output, tc_after_encoder,
    #                                          tc_after_encoder)
    result_counter = 0
    for test_ID in range(number_of_cv_sets):
        test_ID = test_ID + args.start_CV
        if test_ID == 17:
            ntest_samples_perclass_SZ = 7
            ntest_samples_perclass_HC = 15

            sub_per_class_SZ = 144
            sub_per_class_HC = 128

        print('test Id =', test_ID)

        test_start_index_SZ = test_indices_SZ[test_ID]
        test_start_index_HC = test_indices_HC[test_ID]
        test_end_index_SZ = test_start_index_SZ + ntest_samples_perclass_SZ
        test_end_index_HC = test_start_index_HC + ntest_samples_perclass_HC
        total_HC_index_tr_val = torch.cat([HC_index[:test_start_index_HC], HC_index[test_end_index_HC:]])
        total_SZ_index_tr_val = torch.cat([SZ_index[:test_start_index_SZ], SZ_index[test_end_index_SZ:]])

        HC_index_test = HC_index[test_start_index_HC:test_end_index_HC]
        SZ_index_test = SZ_index[test_start_index_SZ:test_end_index_SZ]

        total_HC_index_tr = total_HC_index_tr_val[:(total_HC_index_tr_val.shape[0] - nval_samples_perclass_HC)]
        total_SZ_index_tr = total_SZ_index_tr_val[:(total_SZ_index_tr_val.shape[0] - nval_samples_perclass_SZ)]

        HC_index_val = total_HC_index_tr_val[(total_HC_index_tr_val.shape[0] - nval_samples_perclass_HC):]
        SZ_index_val = total_SZ_index_tr_val[(total_SZ_index_tr_val.shape[0] - nval_samples_perclass_SZ):]

        auc_arr = torch.zeros(ngtrials, 1)
        avg_auc = 0.
        for trial in range(ntrials):
                print ('trial = ', trial)

                g_trial=1
                output_text_file = open(output_path, "a+")
                output_text_file.write("CV = %d Trial = %d\r\n" % (test_ID,trial))
                output_text_file.close()
                # Get subject_per_class number of random values
                HC_random = torch.randperm(total_HC_index_tr.shape[0])
                SZ_random = torch.randperm(total_SZ_index_tr.shape[0])
                HC_random = HC_random[:sub_per_class_HC]
                SZ_random = SZ_random[:sub_per_class_SZ]
                # HC_random = torch.randint(high=len(total_HC_index_tr), size=(sub_per_class,))
                # SZ_random = torch.randint(high=len(total_SZ_index_tr), size=(sub_per_class,))
                #

                # Choose the subject_per_class indices from HC_index_val and SZ_index_val using random numbers

                HC_index_tr = total_HC_index_tr[HC_random]
                SZ_index_tr = total_SZ_index_tr[SZ_random]


                tr_index = torch.cat((HC_index_tr, SZ_index_tr))
                val_index = torch.cat((HC_index_val, SZ_index_val))
                test_index = torch.cat((HC_index_test, SZ_index_test))

                tr_index = tr_index.view(tr_index.size(0))
                val_index = val_index.view(val_index.size(0))
                test_index = test_index.view(test_index.size(0))



                tr_eps = finalData2[tr_index.long(), :, :, :]

                val_eps = finalData2[val_index.long(), :, :, :]
                test_eps = finalData2[test_index.long(), :, :, :]

                # indexx = torch.tensor(np.array([0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15])) # Doing this so I know odd indices is one class and even indices is the other class. Otherwise you can return targets(labels) and see which index is which class. If you do this you also have to make sure to use sequential sampler in trainer (graph_the_works)
                # test_eps = test_eps[indexx.long(), :, :, :]# Doing this so I know odd indices is one class and even indices is the other class. Otherwise you can return targets(labels) and see which index is which class. If you do this you also have to make sure to use sequential sampler in trainer (graph_the_works)



                tr_labels = all_labels[tr_index.long()]
                val_labels = all_labels[val_index.long()]
                test_labels = all_labels[test_index.long()]

                # test_labels = test_labels[indexx.long()]# Doing this so I know odd indices is one class and even indices is the other class. Otherwise you can return targets(labels) and see which index is which class. If you do this you also have to make sure to use sequential sampler in trainer (graph_the_works)



                tr_labels = tr_labels.to(device)
                val_labels = val_labels.to(device)
                test_labels = test_labels.to(device)



                tr_eps = tr_eps.to(device)
                # val_eps = val_eps.to(device)
                test_eps = test_eps.to(device)

                
                print(tr_eps.shape)
                print(val_eps.shape)
                print(test_eps.shape)

                print(tr_labels.shape)
                print(val_labels.shape)
                print(test_labels.shape)

                


                observation_shape = finalData2.shape
                L=""
                lmax=""
                number_of_graph_channels = 1
                if args.model_type == "graph_the_works":
                    print('obs shape',observation_shape[3])
                    encoder = NatureCNN(observation_shape[3], args)
                    encoder.to(device)
                    dir = ""




                print(samples_per_subject)
                complete_model = combinedModel(encoder, PT=args.pre_training, exp=args.exp, device_one=device, oldpath=args.oldpath,n_regions=n_regions,device_two=device2,device_zero=device2,device_extra=device2 )
                complete_model.to(device)
                
                config = {}
                config.update(vars(args))
                # print("trainershape", os.path.join(wandb.run.dir, config['env_name'] + '.pt'))
                config['obs_space'] = observation_shape  # weird hack
                if args.method == "graph_the_works":
                    trainer = the_works_trainer(complete_model, config, device=device2, device_encoder=device,
                                                tr_labels=tr_labels,
                                          val_labels=val_labels, test_labels=test_labels, trial=str(trial),
                                                crossv=str(test_ID),gtrial=str(g_trial))

                else:
                    assert False, "method {} has no trainer".format(args.method)
               
                results[result_counter][0], results[result_counter][1], results[result_counter][2], \
                results[result_counter][3],results[result_counter][4],\
                results[result_counter][5] = trainer.train(tr_eps, val_eps, test_eps) # you can return the weights (final(edge weights) and temporal(temporal_edge_weights)) by editing the code in the train function

                result_counter = result_counter + 1
                tresult_csv = os.path.join(args.path, 'test_results' + sID + '.csv')
                np.savetxt(tresult_csv, results.numpy(), delimiter=",")


    np_results = results.numpy()
    auc = np_results[:,1]
    acc = np_results[:, 0]
    print(np.mean(acc[:]))
    print(np.mean(auc[:]) )

    

    # np_adjacency_matrices = adjacency_matrices_learned.numpy()

    # print('fiinal shape ',temporal_adjacency_matrices_learned.shape)
    # np_temporal_adjacency_matrices = temporal_adjacency_matrices_learned.numpy()
    
    tresult_csv = os.path.join(args.path, 'test_results' + sID + '.csv')
    np.savetxt(tresult_csv, np_results, delimiter=",")
    # with open('../fMRI/Transformer/ICA/FBIRN/position_encoding/temporaladjacencymatrix'+str(JobID)+'.npz', 'wb') as filesim:
    #     np.save(filesim, np_temporal_adjacency_matrices)

    # with open('../fMRI/FBIRN/AdjacencyMatrices/DICE/adjacencymatrix' + str(JobID) + '.npz', 'wb') as filesim:
    #     np.save(filesim, np_adjacency_matrices)


   
    elapsed = time.time() - start_time
    print('total time = ', elapsed)


if __name__ == "__main__":
    CUDA_LAUNCH_BLOCKING = "1"
    # torch.manual_seed(33)
    # np.random.seed(33)
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['pretraining-only']
    config = {}
    config.update(vars(args))
    train_encoder(args)
