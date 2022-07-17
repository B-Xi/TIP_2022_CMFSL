# -*- coding: utf-8 -*-
import numpy as np
import time
import collections
from sklearn import metrics
#import averageAccuracy
def outputRecord(ELEMENT_ACC_RES_SS4, AA_RES_SS4, OA_RES_SS4, KAPPA_RES_SS4,
                 ELEMENT_PRE_RES_SS4, AP_RES_SS4, TRAINING_TIME_RES_SS4, TESTING_TIME_RES_SS4,
                 CATEGORY, ITER, path1):
    print_matrix = np.zeros((CATEGORY * 2 + 6, ITER + 1), dtype=object)
    print_matrix[0:CATEGORY, 0:ITER] = np.around(ELEMENT_ACC_RES_SS4, 4)
    print_matrix[CATEGORY, 0:ITER] = np.around(AA_RES_SS4, 4)
    print_matrix[CATEGORY + 1, 0:ITER] = np.around(OA_RES_SS4, 4)
    print_matrix[CATEGORY + 2, 0:ITER] = np.around(KAPPA_RES_SS4, 4)
    print_matrix[CATEGORY + 3:CATEGORY * 2 + 3, 0:ITER] = np.around(ELEMENT_PRE_RES_SS4, 4)
    print_matrix[CATEGORY * 2 + 3, 0:ITER] = np.around(AP_RES_SS4, 4)
    print_matrix[CATEGORY * 2 + 4, 0:ITER] = np.around(TRAINING_TIME_RES_SS4, 4)
    print_matrix[CATEGORY * 2 + 5, 0:ITER] = np.around(TESTING_TIME_RES_SS4, 4)
    element_mean = np.mean(print_matrix[:, :-1], axis=1)
    element_std = np.std(np.float64(print_matrix[:, :-1]), axis=1)
    for i in range(CATEGORY * 2 + 4):
        print_matrix[i, ITER] = "{:.2f}".format(element_mean[i] * 100) + " ± " + "{:.2f}".format(element_std[i] * 100)
    for i in range((CATEGORY * 2 + 4), (CATEGORY * 2 + 6)):
        print_matrix[i, ITER] = "{:.2f}".format(element_mean[i]) + " ± " + "{:.2f}".format(element_std[i])
    np.savetxt(path1, print_matrix.astype(str), fmt='%s', delimiter="\t", newline='\n')


def outputStats(KAPPA_AE, OA_AE, AA_AE, ELEMENT_ACC_AE, TRAINING_TIME_AE, TESTING_TIME_AE, history, loss_and_metrics, CATEGORY, path1, path2):


    f = open(path1, 'a')

    sentence0 = 'KAPPAs, mean_KAPPA ± std_KAPPA for each iteration are:' + str(KAPPA_AE) + str(np.mean(KAPPA_AE)) + ' ± ' + str(np.std(KAPPA_AE)) + '\n'
    f.write(sentence0)
    sentence1 = 'OAs, mean_OA ± std_OA for each iteration are:' + str(OA_AE) + str(np.mean(OA_AE)) + ' ± ' + str(np.std(OA_AE)) + '\n'
    f.write(sentence1)
    sentence2 = 'AAs, mean_AA ± std_AA for each iteration are:' + str(AA_AE) + str(np.mean(AA_AE)) + ' ± ' + str(np.std(AA_AE)) + '\n'
    f.write(sentence2)
    sentence3 = 'Total average Training time is :' + str(np.sum(TRAINING_TIME_AE)) + '\n'
    f.write(sentence3)
    sentence4 = 'Total average Testing time is:' + str(np.sum(TESTING_TIME_AE)) + '\n'
    f.write(sentence4)

    element_mean = np.mean(ELEMENT_ACC_AE, axis=0)
    element_std = np.std(ELEMENT_ACC_AE, axis=0)
    sentence5 = "Mean of all elements in confusion matrix:" + str(np.mean(ELEMENT_ACC_AE, axis=0)) + '\n'
    f.write(sentence5)
    sentence6 = "Standard deviation of all elements in confusion matrix" + str(np.std(ELEMENT_ACC_AE, axis=0)) + '\n'
    f.write(sentence6)

    f.close()

    print_matrix = np.zeros((CATEGORY), dtype=object)
    for i in range(CATEGORY):
        print_matrix[i] = str(element_mean[i]) + " ± " + str(element_std[i])

    np.savetxt(path2, print_matrix.astype(str), fmt='%s', delimiter="\t",
               newline='\n')

    print('Test score:', loss_and_metrics[0])
    print('Test accuracy:', loss_and_metrics[1])
    print(history.history.keys())


def outputStats_assess(KAPPA_AE, OA_AE, AA_AE, ELEMENT_ACC_AE, CATEGORY, path1, path2):


    f = open(path1, 'a')

    sentence0 = 'KAPPAs, mean_KAPPA ± std_KAPPA for each iteration are:' + str(KAPPA_AE) + str(np.mean(KAPPA_AE)) + ' ± ' + str(np.std(KAPPA_AE)) + '\n'
    f.write(sentence0)
    sentence1 = 'OAs, mean_OA ± std_OA for each iteration are:' + str(OA_AE) + str(np.mean(OA_AE)) + ' ± ' + str(np.std(OA_AE)) + '\n'
    f.write(sentence1)
    sentence2 = 'AAs, mean_AA ± std_AA for each iteration are:' + str(AA_AE) + str(np.mean(AA_AE)) + ' ± ' + str(np.std(AA_AE)) + '\n'
    f.write(sentence2)

    element_mean = np.mean(ELEMENT_ACC_AE, axis=0)
    element_std = np.std(ELEMENT_ACC_AE, axis=0)
    sentence5 = "Mean of all elements in confusion matrix:" + str(np.mean(ELEMENT_ACC_AE, axis=0)) + '\n'
    f.write(sentence5)
    sentence6 = "Standard deviation of all elements in confusion matrix" + str(np.std(ELEMENT_ACC_AE, axis=0)) + '\n'
    f.write(sentence6)

    f.close()

    print_matrix = np.zeros((CATEGORY), dtype=object)
    for i in range(CATEGORY):
        print_matrix[i] = str(element_mean[i]) + " ± " + str(element_std[i])

    np.savetxt(path2, print_matrix.astype(str), fmt='%s', delimiter="\t",
               newline='\n')


def outputStats_SVM(KAPPA_AE, OA_AE, AA_AE, ELEMENT_ACC_AE, TRAINING_TIME_AE, TESTING_TIME_AE, CATEGORY, path1, path2):


    f = open(path1, 'a')

    sentence0 = 'KAPPAs, mean_KAPPA ± std_KAPPA for each iteration are:' + str(KAPPA_AE) + str(np.mean(KAPPA_AE)) + ' ± ' + str(np.std(KAPPA_AE)) + '\n'
    f.write(sentence0)
    sentence1 = 'OAs, mean_OA ± std_OA for each iteration are:' + str(OA_AE) + str(np.mean(OA_AE)) + ' ± ' + str(np.std(OA_AE)) + '\n'
    f.write(sentence1)
    sentence2 = 'AAs, mean_AA ± std_AA for each iteration are:' + str(AA_AE) + str(np.mean(AA_AE)) + ' ± ' + str(np.std(AA_AE)) + '\n'
    f.write(sentence2)
    sentence3 = 'Total average Training time is :' + str(np.sum(TRAINING_TIME_AE)) + '\n'
    f.write(sentence3)
    sentence4 = 'Total average Testing time is:' + str(np.sum(TESTING_TIME_AE)) + '\n'
    f.write(sentence4)

    element_mean = np.mean(ELEMENT_ACC_AE, axis=0)
    element_std = np.std(ELEMENT_ACC_AE, axis=0)
    sentence5 = "Mean of all elements in confusion matrix:" + str(np.mean(ELEMENT_ACC_AE, axis=0)) + '\n'
    f.write(sentence5)
    sentence6 = "Standard deviation of all elements in confusion matrix" + str(np.std(ELEMENT_ACC_AE, axis=0)) + '\n'
    f.write(sentence6)

    f.close()

    print_matrix = np.zeros((CATEGORY), dtype=object)
    for i in range(CATEGORY):
        print_matrix[i] = str(element_mean[i]) + " ± " + str(element_std[i])

    np.savetxt(path2, print_matrix.astype(str), fmt='%s', delimiter="\t",
               newline='\n')