'''
Configuration of the project (used by robustness.py). 
Also provides classes implementing priority vectors, perturbations and methods.

Limitations:
Supports only two classes of the distribution of perturbations (Normal and Skewed Normal).

Careful:
All methods (EVM, GMM, COH) are implemented assuming the Additive model! 
That is, reciprocal PCMs are with 0 on the diagonal, and PVs, if normalized, are adjusted such that sum(PV) = 0.

Copyright 2023 Jan Gorecki
'''

from scipy.stats import norm, skewnorm
import numpy as np
from abc import ABC, abstractmethod



class PriorityVector(np.ndarray):
    '''Container for priorities'''
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls) # assure that an instance of PriorityVector is returned (rather than of ndarray)
        return obj


    def rank(self):
        # returns rank ordering of the PV
        return np.argsort(np.argsort(self))


    def are_weights_unique(self):
        # given vector pv, checks if all its elemenets are different, and returns True if so, False otherwise
        unique = len(set(self)) == len(self)
        return unique


    def get_consistent_pcm(self):
        # Returns a PCM consistent with the priority vector (self)
        n = len(self)
        pcm = np.zeros((n, n))
        for i in range(n):
            for j in range(i+1, n): # iterate just above the diagonal
                pcm[i, j] = self[i] - self[j]
                pcm[j, i] = -pcm[i, j]          # reciprocity
        return pcm



class Perturbation:
    '''Model for random perturbations'''
    def __init__(self, name, sigma, loc = 0, a = None, force_zero_mean = False):
        '''
        name                - name of the distribution modeling the perturbations (in ['Normal', 'Skewnorm'])
        sigma               - spread of the perturbations
        loc                 - location of the perturbations
        a                   - skewness of the perturbations
        force_zero_mean     - by setting to True, zero mean of the perturbations is assured
        '''
        self.name = name
        assert self.name in ['Normal', 'Skewnorm'], "name NOT in ['Normal', 'Skewnorm']!"
        if self.name == 'Normal':
            self.sigma = sigma # uncertainty of the decision maker -  std of normal distr for perturbation
            self.loc = loc # mean of perturbation
            if force_zero_mean and loc != 0:
                raise Exception(f'loc != 0 but {force_zero_mean=}!')
        else: # name == 'Skewnorm':
            self.a = a # skewness
            self.sigma = sigma # uncertainty of the decision maker  
            if force_zero_mean:# assure mean = 0?
                mean = skewnorm.stats(a, loc = loc, scale = sigma, moments='m')
                self.loc = -mean # center the distribution, i.e., mean = 0
                print(f'loc shifted from {loc} to {self.loc}')
            else: # mean if free
                self.loc = loc


    def fname(self):
        sigma = f"sigma_{str(self.sigma).replace('.','_')}"
        if self.name == 'Normal':
            return sigma
        else:
            return f"a_{str(self.a).replace('.','_')}_" + sigma 


    def repr(self):
        return f"{self.n}_N_{self.N}_{self.name}_" + self.fname()


    def plot_pdf(self, ax, ls):
        a = 0 if self.name == 'Normal' else self.a # norm is skewnorm with a = 0
        x = np.linspace(-5, 5, 50)
        ax.plot(x, skewnorm.pdf(x, a, loc = self.loc, scale = self.sigma),  'brown', ls = ls, lw=2, alpha=0.6, label='skewnorm pdf')


    def rnd(self):
        if self.name == 'Normal': 
            return norm.rvs(loc = self.loc, scale = self.sigma, size = 1)
        else: # 'Skewnorm' 
            return skewnorm.rvs(self.a, loc = self.loc, scale = self.sigma, size = 1)


    def eval_sf(self, value): # 
        # evaluate the survival function (sf = 1-cdf) if the perturbation
        if self.name == 'Normal':
            return norm.sf(value, loc = self.loc, scale = self.sigma) # formula from the paper 
        else:
            return skewnorm.sf(value, self.a, loc = self.loc, scale = self.sigma) 



class Method(ABC):
    ''' Template for methods that calculate the rank ordering of a PV for a reciprocal PCM in the additive model.
    In the paper notation, r(M(A)) is calculated by M.pcm2ranks(A), where M is an instance of Method.
    '''
    
    def __init__(self, name, color):
        self.name = name     # abbrevitation
        self.color = color   # color in plots


    @abstractmethod
    def pcm2ranks(self, pcm):
        pass


    def __repr__(self):
        return self.name



class SaatyEigenvectorMethod(Method):

    RI = {3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24} # Saaty's Average Random Index for n = 3,...,6

    def __init__(self):
        super().__init__('EVM', 'darkorange')         


    def pcm2ranks(self, pcm):
        # pcm is assumed to be in additive model
        mult_pcm = np.exp(pcm) # strictly positive matrix
        eig_values, eig_vectors = np.linalg.eig(mult_pcm)
        i_lambda_max = np.argmax(eig_values)
        mult_w = PriorityVector(eig_vectors[:, i_lambda_max].real) # eigenvector corrsponding to lambda_max
                                                   # note that as mult_pcm is > 0, we know that all elements
                                                   # of the eigenvector corresponding to the largest eigenvalue are real of the same sign
        if mult_w[0] < 0:
            mult_w = -mult_w # make all elementes positive (as we know that now all have to be negative!)
        assert (mult_w > 0).all(), 'mult_w > 0 violated!'
        return mult_w.rank() # due to the monotonic isomorphism exp(), the we don't need, to waste time by transforming back to the additive model


    def CR(self, pcm):
        # compute consistency ratio for the given PCM
        mult_pcm = np.exp(pcm) # strictly positive matrix
        eig_values, _ = np.linalg.eig(mult_pcm)
        lambda_max = eig_values.max().real
        n = pcm.shape[0]
        CI = max(0., (lambda_max - n) / (n - 1))
        RI = SaatyEigenvectorMethod.RI[n]
        return CI/RI



class GeometricMeanMethod(Method):
    def __init__(self):
        super().__init__('GMM', 'dodgerblue') 


    def pcm2ranks(self, pcm):
        # geometric mean method for the additive model, i.e, v_i = (sum_j=1...n (a_ij)) / n
        # NOTE: v is inherently normalized (note that sum(pcm) = 0 due to reciprocity, 
        #       hence sum of elements of v must also equal to 0)
        return PriorityVector(np.sum(pcm, axis = 1) / pcm.shape[0]).rank()



class CoherencyMethod(Method):
    def __init__(self):
        super().__init__('COH', 'black') 


    def pcm2ranks(self, pcm):
        ranks = (pcm > 0).sum(axis = 1) # sum of indicators - NOTE: due to Python standards, we don't add +1 to the ranks and start counting from 0
        if not PriorityVector(ranks).are_weights_unique():
            ranks = None # the method does not produce any output if the pcm is not coherent, which is detected
                     # by the uniqueness of ranks
        return ranks



# methods to include in the comparison
methods = [SaatyEigenvectorMethod(),
           GeometricMeanMethod(),
           CoherencyMethod()]


class Cfg:
    ''' Project configuration'''
    def __init__(self, N, do_random_v_space, zero_mean_exp, coherent_PCMs_only, load_results, save_fig, plot_type): 
        '''
        N                   - number of replications
        do_random_v_space   - if False, compute the v-robustness. If true, compute the robustness
        zero_mean_exp       - True, to replicate the zero mean perturbation results, False for non-zero mean perturbation results
        coherent_PCMs_only  - if True, only coherent perturbed PCMs are considered (in order to involve the COH method into the comparison),
                              if False, all perturbed PCMs are considered (and thus the COH method is excluded from the comparison)
        load_results        - if False, compute new results and save. If True, just load previously saved results.
        save_fig            - if False, just show the figure without saving, if True, save the figure without showing
        '''

        # methods involved in the comparison
        self.methods = methods

        self.N = N
        self.do_random_v_space = do_random_v_space
        self.zero_mean_exp = zero_mean_exp
        self.coherent_PCMs_only = coherent_PCMs_only
        if not coherent_PCMs_only:
            # remove the last method (COH) as cannot guarantee that we get 
            # an ouput from this method for a given perturbed PCM
            self.methods = methods[:-1]
            assert do_random_v_space, 'coherent_PCMs_only = False supported only for do_random_v_space = True!'
        self.load_results = load_results
        self.save_fig = save_fig
        assert plot_type in ['analytical', 'numerical', None], "plot_type NOT in ['analytical', 'numerical', None]!" # None serves for the analytical plot
        self.plot_type = plot_type

        if not self.do_random_v_space:
            self.ns = [3] # PCM dimesions
            self.sigmas = [1/2, 1, 2]
            # non-random 1D v2space
            if self.zero_mean_exp:
                self.aas = [-5, 0, 5] # skewness parameters
            else: # non zero_mean_exp
                self.locs = [-1, -1/2, 1/2, 1]
        else:
            self.ns = range(3, 7) # PCM dimensions
            self.sigmas = [0.05, 0.1, 0.2, 0.5, 1]
            if self.zero_mean_exp:
                self.aas = [0] # just N(0, sigma^2)
            else: # non zero_mean_exp
                raise Exception('Not considered in our work.')


    def get_perturbations_model(self, sigma, a = None, loc = None):
        if self.zero_mean_exp:
            assert a is not None, 'a is None!'
            return Perturbation('Skewnorm', sigma, a = a, force_zero_mean = True)
        else:
            assert loc is not None, 'loc is None!'
            return Perturbation('Normal', sigma, loc = loc, force_zero_mean = False)