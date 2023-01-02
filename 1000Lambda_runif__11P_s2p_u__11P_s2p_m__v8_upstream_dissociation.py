from ICIER_extended_v8 import *

print('start:\t' + str(datetime.datetime.now()), flush=True)

condition_list = []

with open('Lambda_input/num_runif.txt', 'r') as r_in_file:
    r_in_lines = r_in_file.readlines()

for i in [k / 10 for k in range(0, 11)]:
    for j in [k / 10 for k in range(0, 11)]:
        for k in r_in_lines:
            tmp = default_condition_set.copy()

            tmp['P_clear_up'] = 1
            tmp['P_clear_down'] = 0  # set as upstream dissociation mode

            tmp['Lambda'] = float(k.split('\t')[1])
            tmp['P_s2p_u'] = i
            tmp['P_s2p_m'] = j
            condition_list.append(tmp)


Sim_multi_condition(condition_list, 'sim_result/1000Lambda_runif__11P_s2p_u__11P_s2p_m__v8_upstream_dissociation.txt')

print('start:\t' + str(datetime.datetime.now()), flush=True)