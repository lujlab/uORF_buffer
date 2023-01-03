# uORF_buffer

This repository stores the scripts used for simulation of the translation process of an mRNA molecule. The model used here is based on Andreev et al.'s ICIER model [(ref)](https://elifesciences.org/articles/32563), which is distributed under GPLv3 licence at the following address: [https://github.com/maximarnold/uORF_TASEP_ICIER](https://github.com/maximarnold/uORF_TASEP_ICIER). The original model is implemented in MATLAB, and we adpated it using python.


## Usage of the scripts
The core components of the model is integrated into the file [ICIER_extended_v8.py](https://github.com/lujlab/uORF_buffer/blob/main/ICIER_extended_v8.py). The function `Sim_main(condition_set)` executes a single run of simulation, while the function `Sim_multi_condition(condition_list, outfile_path)` executes multiple runs of simulation.
