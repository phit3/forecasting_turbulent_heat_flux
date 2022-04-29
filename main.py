#!/usr/bin/python3
import argparse
import os
import re
import json
import matplotlib.pyplot as plt
import math

from model_manager import ModelManager
from fr import FR

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replication package for the GRU experiments of: Direct data-driven forecast of local turbulent heat flux in Rayleigh–Bénard convection.')
    parser.add_argument('-o', '--operation', dest='operation', default='test', choices=['train', 'predict'], help='operation that shall be performed.')
    parser.add_argument('-q', '--quiet', dest='quiet', default=False, action='store_true', help='reduces noise on your command line.')

    args = parser.parse_args()
    operation = args.operation
    quiet = args.quiet
   
    if operation == 'train':
        split = (0.89, 0.11, 0)
        dataset = 'Pr07_uytheta_40_trainval'
        hyperparameters = {'filename': 'Pr07_uytheta_40_trainval',
                           'dimensions': 40,
                           'lr': 1e-3,
                           'gamma': 0.6,
                           'plateau': 10,
                           'latent_dim': 384,
                           'output_steps': 100}
    elif operation == 'predict':
        split = (0, 0, 1)
        dataset = 'Pr07_uytheta_40_test'
        hyperparameters = {'filename': 'Pr07_uytheta_40_test',
                           'dimensions': 40,
                           'lr': 1e-3,
                           'gamma': 0.6,
                           'plateau': 10,
                           'latent_dim': 384,
                           'output_steps': 900}

    filename = hyperparameters['filename']

    checkpoint_dir = 'results/checkpoints'

    if operation == 'train' or os.path.isfile(os.path.join(checkpoint_dir, 'encoder')) and os.path.isfile(
            os.path.join(checkpoint_dir, 'decoder')):
        print('Running {} operation for {} on {}.'. format(operation, 'FR', dataset))
    else:
        print(f'[ERROR] Could not find required checkpoints in {checkpoint_dir}.')
        exit(1)
    try:
        mgr = ModelManager(model_class=FR, dataset=filename, split=split, quiet=quiet)
        if operation == 'train':
            mgr.train_model(override_args=hyperparameters)
        elif operation == 'predict':
            predictions, realities = mgr.predict_model(override_args=hyperparameters)
            fs = 5
            fig, axs = plt.subplots(fs, figsize=(15, 15))
            for i in range(0, fs):
                p = predictions[:, i]
                r = realities[:, i]
                axs[i].plot(realities[:, i, 0])
                axs[i].plot(predictions[:, i, 0])
                axs[i].set_xlabel('timestep')
                axs[i].set_ylabel('$z^{(' + str(i) + ')}$')
                axs[0].legend(['ground truth', 'prediction'])
            plt.show()
    except Exception as e:
        error_type = type(e).__name__
        print(f'{error_type}: {e}')
