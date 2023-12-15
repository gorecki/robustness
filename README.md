# Robustness
*Unifying Python framework for evaluating robustness of pairwise comparison methods against rank reversal*
 
Just run robustness.py (depends on config.py) in order to replicate the results from
"Jan Górecki, David Bartl and Jaroslav Ramík (2023).  Robustness of priority deriving methods for
pairwise comparison matrices against rank reversal: A probabilistic approach. *Annals of Operations Research*. DOI: https://doi.org/10.1007/s10479-023-05753-0"

If one does not want to compute all the results but wants to replicate the plots,
just switch the value of argument load_results to True in replicate(), and the precomputed 
results from the folder *results* will be used. 
