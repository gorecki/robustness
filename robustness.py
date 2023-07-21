'''
Unifying framework for evaluating robustness of pairwise comparison methods against rank reversal

Implementing the framework developed in:
Jan Gorecki, David Bartl and Jaroslav Ramik (2023). Robustness of methods calculating priority 
vectors from pairwise comparison matrices against rank reversal: A probabilistic concept
(below referred to as 'the paper')

Copyright 2023 Jan Gorecki
'''

import numpy as np
import matplotlib.pyplot as plt
import pickle
from time import time
import os

from config import Cfg, PriorityVector



def add_perturbations(pcm, perturbation):
    # Add perturbations to a consistent PCM pcm
    perturbed_pcm = pcm.copy()
    n = pcm.shape[0]
    for i in range(n):
        for j in range(i+1, n): # iterate just above the diagonal
            X_ij = perturbation.rnd()                   # generate random perturbation
            perturbed_pcm[i, j] = pcm[i, j] + X_ij      # add it to the consistent matrix 
            perturbed_pcm[j, i] = -perturbed_pcm[i, j]  # keep reciprocity
    return perturbed_pcm


def analytical_COH_robustness(pvec, perturbation):
    # Computes the v-robustness of the COH method using the analytical
    # approach developed in the paper (only for n = 3)
    assert pvec.are_weights_unique(), f'weights in {pvec=} are not unique!'
    assert pvec[0] > pvec[1] > pvec[2], 'assumption of pvec[0] > pvec[1] > pvec[2] violated!'
    pcm = pvec.get_consistent_pcm()
    r = 1
    n = pvec.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            assert pcm[i, j] > 0, 'pcm[i, j] <= 0!'
            r *= perturbation.eval_sf(-pcm[i, j])
    return r 


def create_dir_if_not_exists(dir_name):
    # Check if the directory exists abd if not, create it
    if not os.path.exists(dir_name):
        # Create the directory if it does not exist
        os.makedirs(dir_name)
        print(f"Directory {dir_name} created successfully")


###### Monte Carlo simulations #############################


def replication_coh(PVgenerator, cfg, perturbation):
    # Perform one replication of the MC approach developed in the paper.
    # For all methods involved in cfg, computes if there is a rank reversal 
    # after perturbations to the consistent PCM are added 
    results = {'n_all_reps': 0}
    for method in cfg.methods:
        results['n_equal_ranks_' + method.name] = 0 
        results['n_' + method.name] = 0
    do_rep = True
    # repeat until a coherent noisy PCM if generated (computing the v-robustness conditioned on that the 
    # generated PCM is coherent)
    while do_rep:
        pv = PVgenerator()             # generate the "uknown" PV 
        if not pv.are_weights_unique(): # uniqueness check
            continue # some of weights are duplicate, so repeat     
        pcm = pv.get_consistent_pcm()                            # generate consistent PCM from random vector
        noisy_pcm = add_perturbations(pcm, perturbation)         # add some noise to that consistent PCM

        ranks = {} # ranks for each method
        for method in cfg.methods:
            ranks[method.name] = method.pcm2ranks(noisy_pcm)

        # check uniqueness of outputted ranks (except for COH)
        all_unique = True
        for method in cfg.methods:
            if method.name != 'COH': # we don't test COH as it is either unique or None
                if not ranks[method.name].are_weights_unique():
                    all_unique = False
                    print(f'{method.name} generated non-unique ranks {ranks[method.name]}')
        if not all_unique:
            continue # some output was non-unique, so repeat the replication

        results['n_all_reps']  += 1 # we have obtained unique PVs from all methods (note that COH could return None, which is Ok)

        if ranks['COH'] is None:
            # the noisy_pcm is not coherent, so repeat the replication
            continue

        # now, we know that noisy_pcm is coherent, so we can do all statistics 
        true_ranks = pv.rank()
        for method in cfg.methods:
            results['n_' + method.name] += 1 # count the method outputs
            results['n_equal_ranks_' + method.name] += (ranks[method.name] == true_ranks).all() # +1 if all ranks match, +0 otherwise

        do_rep = False # all statistics done, so we don't further repeat the replication

    return results


def exp_core(PVgenerator, cfg, perturbation):
    # Repeat replication_coh() cfg.N times and sum up the results
    results = {'n_all_reps': 0}
    for method in cfg.methods:
        results['n_equal_ranks_' + method.name] = 0 
        results['n_' + method.name] = 0
    for i in range(cfg.N):
        i_results = replication_coh(PVgenerator, cfg, perturbation) # jen na coherent 
        for method in cfg.methods:
            results['n_equal_ranks_' + method.name] += i_results['n_equal_ranks_' + method.name]
            results['n_' + method.name] += i_results['n_' + method.name]
        results['n_all_reps'] += i_results['n_all_reps']
    return results



def exp_prob(cfg, perturbation, n, v1, v2_space, v3):
    # iterates of the space of pririty vectors (represented here by v1, v2_space, v3)
    # and computes the robustness via our MC apprach and eventually also via our analytical 
    # approach for the COH method
    robustness = {}
    for rob_type in ['rough']:
        for method in cfg.methods:
            robustness[f'cond_{rob_type}_' + method.name] = []
    robustness.update({'rough_COH': [],
                       'rough_analyt_COH': [], # we want to order the keys like this
                       'COH_ratio': []})
    if cfg.do_random_v_space: 
        del robustness['rough_analyt_COH'] # we don't have the ordering v3 > v2 > v1 so we don't use the analytic formula
    for v2 in v2_space:
        #print(v2)
        if cfg.do_random_v_space:
            exp_prob_PVgenerator = lambda : PriorityVector(np.random.uniform(low = v1, high = v3, size = n))
        else: #non-random v_space
            exp_prob_PVgenerator = lambda : PriorityVector(np.concatenate((v3, v2, v1), axis=None)) # v3 > v2 > v1 to satisfy the assumptions for analytical_COH_robustness()
        
        results = exp_core(exp_prob_PVgenerator, cfg, perturbation)
        
        for method in cfg.methods:
            robustness['cond_rough_' + method.name].append(results['n_equal_ranks_' + method.name]/results['n_' + method.name])   # rough robustness conditioned on isCOH
        robustness['rough_COH'].append(results['n_equal_ranks_COH']/results['n_all_reps'])                # numerical equivalent to the analalytical case
        if 'rough_analyt_COH' in robustness:
            robustness['rough_analyt_COH'].append(analytical_COH_robustness(exp_prob_PVgenerator(), perturbation))
        robustness['COH_ratio'].append(cfg.N / results['n_all_reps']) # #coherent/#all
        #COHdebug(v)

    print(f"n_all_reps = {results['n_all_reps']}") # just for the last v2_space point
    return robustness


########### plots ##############################


def plot_3D_one_type_R(robustness, cfg, v1, v2_space, v3):
    # creates figures showing the v-robustness computed via our MC approach for n = 3
    n = 3
    robustness = robustness[n] # robustness is computed just for n = 3, so take just these results

    first_sigma = list(robustness.keys())[0] # used to access all a NOTE: set of a is the same for all sigma!!
    n_col = len(robustness[first_sigma]) # each param a in a separate column
    scale = 3
    fig, axs = plt.subplots(3, n_col, figsize=(n_col * scale, 3 * scale))
    sigma_line_style = ('-', '-.', '--')
    legend = []
    legend_rat = []
    for i_a, a in enumerate(robustness[first_sigma]):
        for i_sigma, sigma in enumerate(robustness):
            for method in cfg.methods:
                ax = axs[0, i_a]
                ax.plot(v2_space, robustness[sigma][a]['cond_rough_' + method.name], ls = sigma_line_style[i_sigma], 
                        color = method.color, alpha = 1, lw = 2 if method.name == 'EVM' else 1)
                if i_a == 0:
                    legend.append(f'{method.name} ' r'$\sigma$=' + f'{sigma}')
            # coherence ratio
            ax = axs[1, i_a]
            ax.plot(v2_space, robustness[sigma][a]['COH_ratio'], ls = sigma_line_style[i_sigma], color = 'm', alpha = 0.6)
            if i_a == 0:
                legend_rat.append(r'$\sigma$=' + f'{sigma}')
            perturbation = cfg.get_perturbations_model(sigma, a = a, loc = a) # NOTE (on loc = a): for zero_mean_exp = False, "a" is in fact "loc"
            perturbation.plot_pdf(axs[2, i_a], sigma_line_style[i_sigma]) 

        for i in [0, 1]: # iterate one column
            ax = axs[i, i_a]
            ax.set_xlim((v1, v3))
            ax.set_ylim((0, 1.01))
            ax.set_xlabel('$v_2$')
            if i == 0:
                title = f'{a=}' if cfg.zero_mean_exp else f'$\mu$={a}' # the same small hack
                ax.set_title(title)
    axs[0, 0].legend(legend, fontsize = 6)
    axs[1, 0].legend(legend_rat)
    axs[1, 1].set_title((' ' * (0 if cfg.zero_mean_exp else 55)) + 'Coherency ratio\n')
    axs[2, 0].legend(legend_rat)
    axs[2, 1].set_title((' ' * (0 if cfg.zero_mean_exp else 55)) + 'Perturbations PDF')
    fig.suptitle('(1, $v_2$, 0)-robustness')
    fig.tight_layout()


def plot_nD_one_type_R(robustness, cfg):
    # creates figures showing the robustness computed via our MC approach 
    size = 3
    fig, axes = plt.subplots(1, len(robustness.keys()), figsize=(len(robustness.keys()) * size, size + 2)) 
    x = np.arange(len(cfg.sigmas))  # the label locations
    width = 0.8/len(cfg.methods)
    for i_n, n in enumerate(robustness): # n in ns
        ax = axes[i_n]
        rects = {}
        for i_method, method in enumerate(cfg.methods):
            r = [] # mean of all robustnesses computed on all random Vs
            for r_sigma in robustness[n].values():
                r.append(np.mean(r_sigma[0]['cond_rough_' + method.name]))
            ax.bar(x + (i_method - 1) * width, r, width, color = method.color, label = method.name)

        if i_n == 0:
            ax.set_ylabel('Robustness')
        ax.set_title(f'{n=}')
        ax.set_xticks(x)
        ax.set_xticklabels(cfg.sigmas)
        ax.set_xlabel(r'$\sigma$')
        ax.set_ylim((0, 1))
        if i_n == 0:
            ax.legend()
    fig.tight_layout()


def plot_3D_analyt_COH(cfg):
    # creates the figure including the v-robustness accessed analytically for the COH method
    n = 3
    size = 4 # plot size
    fig, ax = plt.subplots(1, 1, figsize=(size,size))
    N_line_style = {100: ('magenta', ':'),
                    1000: ('orange', '-.'),
                    10000: ('green', '--'),
                    'analytic': ('black', '-')}
    for N, (color, line_style) in N_line_style.items():
        if type(N) == int:
            assert cfg.zero_mean_exp == True, "this plot serves just for cfg.zero_mean_exp = True exps"
            zero_mean_exp_f_addon = f'_ZME_{cfg.zero_mean_exp}'
            do_random_v_space_f_addon = f'_randv2_{cfg.do_random_v_space}'
            res_fname = f'results/res_N_{N}{zero_mean_exp_f_addon}{do_random_v_space_f_addon}.bin'
            with open(res_fname, 'rb') as handle:
                robustness, v1, v2_space, v3 = pickle.load(handle)    
            robustness = robustness[n] # robustness is computed just for n = 3, so take just these results
            method_name = 'rough_COH'
        else:
            # we already loaded file with the analytical results
            method_name = 'rough_analyt_COH'

        sigma = 0.5
        a = 0 # Normal distribution
        if method_name == 'rough_analyt_COH':
            ax.plot(v2_space, robustness[sigma][a][method_name], ls = '-', color = 'k', lw = 1.5, label = 'analytic')
        else:
            ax.scatter(v2_space, robustness[sigma][a][method_name], ls = '-', color = color, marker = 'o', alpha = 0.6, label = f'{N=}')
    ax.set_xlim((v1, v3))
    ax.set_ylim((0, 1.01))
    ax.set_title(f'Analytical vs numerical\nrobustness of COH')
    ax.set_xlabel('$v_2$')
    ax.set_ylabel('Robustness of $(1, v_2, 0)$')
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [0,1,2,3]
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order])
    #ax.legend(legend) 
    fig.tight_layout()


###### Simulation wrappers ##################################


def exp_prob_inner_wrapper(cfg, n, v1, v2_space, v3):
    # For the given dimension of PCMs (n), iterates over all parameters
    # of the distributions of perturbations and for each one computes 
    # an estimate of the robustness using the MC approach developed in the paper
    robustness = {} # all results
    for sigma in cfg.sigmas:
        robustness[sigma] = {}
        second_params = cfg.aas if cfg.zero_mean_exp else cfg.locs # iterate either over aas or over locs
        for second_param in second_params:
            perturbation = cfg.get_perturbations_model(sigma, a = second_param, loc = second_param)
            print(f'Computing {sigma=} {second_param=} ...')
            robustness[sigma][second_param] = exp_prob(cfg, perturbation, n, v1, v2_space, v3)
    return robustness


def exp_prob_wrapper(cfg):
    # Main procedure that computes the results for a given cfg
    # Iterate over all dimensions (sizes) of PCMs (cfg.ns) and
    # computes the robustness for all parametrizations of the perturbations
    # Once done, a visualization is shown or saved.
    zero_mean_exp_f_addon = f'_ZME_{cfg.zero_mean_exp}'
    do_random_v_space_f_addon = f'_randv2_{cfg.do_random_v_space}'
    res_dir = "results"
    create_dir_if_not_exists(res_dir)
    res_fname = f'{res_dir}/res_N_{cfg.N}{zero_mean_exp_f_addon}{do_random_v_space_f_addon}.bin'
    if cfg.load_results:
        with open(res_fname, 'rb') as handle:
            robustness, v1, v2_space, v3 = pickle.load(handle)    
    else: # compute results
        t_start = time()
        v1 = 0
        v3 = 1
        robustness = {} # robustness over all ns
        np.random.seed(0) # replicability 
        for n in cfg.ns:
            print(f'{n=}')
            if not cfg.do_random_v_space: 
                v2_space = np.linspace(v1 + 0.01, v3 - 0.01, 21)
            else: # cfg.do_random_v_space = True
                v2_space = [None] # None here represents a random v point 
            robustness[n] = exp_prob_inner_wrapper(cfg, n, v1, v2_space, v3) 
        print(f'Computed in {int(time()-t_start)} s')
        # save results
        with open(res_fname, 'wb') as handle:
            pickle.dump((robustness, v1, v2_space, v3), handle, protocol=pickle.HIGHEST_PROTOCOL)

    # plots
    if cfg.plot_type: 
        if cfg.plot_type == 'numerical':
            if cfg.do_random_v_space:
                plot_nD_one_type_R(robustness, cfg)
                fig_name = f'probnD_COH_cond'
            else: # non random v2space
                plot_3D_one_type_R(robustness, cfg, v1, v2_space, v3)
                fig_name = 'prob3D_COH_cond'
        else: # cfg.plot_type == 'analytical'
            plot_3D_analyt_COH(cfg)
            fig_name = 'prob3D_COH_analyt'
        # save or show
        if cfg.save_fig:
            # Define the directory name
            dir_name = "figures"
            create_dir_if_not_exists(dir_name)
            fname = f"{dir_name}/{fig_name}{zero_mean_exp_f_addon}"
            plt.savefig(fname, dpi = 400)
            print(f'{fname} saved.')
        else:
            plt.show(block = True)

'''
TODO: tohle vratit, az to budu publikovat
def replicate():
    # Switch to True in order to replicate the desired result(s)
    to_replicate = {'Figure1': True, 
                    'Figure2': True,
                    'Figure3': True,
                    'Figure4': True}

    if to_replicate['Figure1']:
        for N in [100, 1000, 10000]:
            cfg = Cfg(N = N, do_random_v_space = False, zero_mean_exp = True, 
                      load_results = False, save_fig = True, plot_type = 'analytical' if N == 10000 else None) 
            # NOTE: (explaining plot_type) we first compute the results for N in [100, 1000] without showing the plot, and 
            #       then for N = 10000 we compute the results as well as we show the plot 
            exp_prob_wrapper(cfg) # Takes roughly 10 minutes to compute (on Intel(R) Core(TM) i7-7700 CPU @3.60GHz and 32GB RAM)
                                  

    if to_replicate['Figure2']:
        cfg = Cfg(N = 10000, do_random_v_space = False, zero_mean_exp = True, 
                    load_results = False, save_fig = True, plot_type = 'numerical') 
        exp_prob_wrapper(cfg) # Takes roughly 10 minutes to compute
                              # To get some results faster, decrease N (and expect less smooth outputs)  
    
    if to_replicate['Figure3']:
        cfg = Cfg(N = 10000, do_random_v_space = False, zero_mean_exp = False,  
                    load_results = False, save_fig = True, plot_type = 'numerical') 
        exp_prob_wrapper(cfg) # Takes roughly 10 minutes to compute

    if to_replicate['Figure4']:
        cfg = Cfg(N = 100000, do_random_v_space = True, zero_mean_exp = True, 
                    load_results = False, save_fig = True, plot_type = 'numerical') 
        exp_prob_wrapper(cfg) # Takes roughly 70 minutes to compute
'''

# TODO: urceno jen pro testovani, pak smazat
def replicate():
    # Switch to True in order to replicate the desired result(s)
    to_replicate = {'Figure1': 0, 
                    'Figure2': 1,
                    'Figure3': 1,
                    'Figure4': 1}

    if to_replicate['Figure1']:
        for N in [100, 1000, 10000]:
            cfg = Cfg(N = N, do_random_v_space = False, zero_mean_exp = True, 
                      load_results = 1, save_fig = 0, plot_type = 'analytical' if N == 10000 else None) 
            # NOTE: (explaining plot_type) we first compute the results for N in [100, 1000] without showing the plot, and 
            #       then for N = 10000 we compute the results as well as we show the plot 
            exp_prob_wrapper(cfg) # Takes roughly 10 minutes to compute (on Intel(R) Core(TM) i7-7700 CPU @3.60GHz and 32GB RAM)
                                  

    if to_replicate['Figure2']:
        cfg = Cfg(N = 10000, do_random_v_space = False, zero_mean_exp = True, 
                    load_results = 1, save_fig = 1, plot_type = 'numerical') 
        exp_prob_wrapper(cfg) # Takes roughly 10 minutes to compute
                              # To get some results faster, decrease N (and expect less smooth outputs)  
    
    if to_replicate['Figure3']:
        cfg = Cfg(N = 10000, do_random_v_space = False, zero_mean_exp = False,  
                    load_results = 1, save_fig = 1, plot_type = 'numerical') 
        exp_prob_wrapper(cfg) # Takes roughly 10 minutes to compute

    if to_replicate['Figure4']:
        cfg = Cfg(N = 100000, do_random_v_space = True, zero_mean_exp = True, 
                    load_results = 1, save_fig = 1, plot_type = 'numerical') 
        exp_prob_wrapper(cfg) # Takes roughly 70 minutes to compute

def three_examples():
    from config import methods
    eps = 0.1
    A = np.array([[0, eps, 2, eps],
                  [-eps, 0, 1, 2],
                 [-2,-1, 0, 4],
                  [-eps,-2,-4, 0]])
    for method in methods:
        print(method.name)
        print(method.pcm2ranks(A))


if __name__ == '__main__':
    #replicate()
    three_examples()