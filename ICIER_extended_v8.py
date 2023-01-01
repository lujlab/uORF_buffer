import multiprocessing
import random
import datetime
from numba import jit

# v2: adjusted kicking rules: only when a 80S runs into a 40S AFTER MOVING will it kick the 40S off.
# v3: output written into stdout in real time.
# v4: rewrite the loop to run faster, and fix an error remained in v2 kicking rules.
# v5: fixed an error of 40S running out of range, and allow 40S to run first.
# v6: fixed an error of moving determination.
# v7: imported numba to run much faster.
# v8: integrated upstream/downstream/double-dissociation model.

default_condition_set = {'Ls': 10,
                         'Le': 10,
                         'Len_a': 50,  # length before uATG
                         'Len_u': 30,  # length between uATG and uSTOP, both ends included
                         'Len_b': 50,  # length between uSTOP and mATG, both ends not included
                         'Len_m': 500,  # length between mATG and mSTOP, both ends included
                         'Len_c': 50,  # length after mSTOP
                         'Lambda': 0.1,
                         # probability of 40S loading in a single action.
                         # CAUTION: in the article, the symbol of this parameter is R_in.
                         'P_delay_au': 1,  # probability NOT to delay at au boundary, 1 means no delay
                         'P_delay_ub': 1,  # probability NOT to delay at ub boundary, 1 means no delay
                         'P_delay_bm': 1,  # probability NOT to delay at bm boundary, 1 means no delay
                         'P_delay_mc': 1,  # probability NOT to delay at mc boundary, 1 means no delay
                         'P_s2p_u': 0.1,
                         # probability of transformation from 40S to 80S at uATG in a single action
                         # CAUTION: in the article, the symbol of this parameter is I_uORF.
                         # final transforming probability at uATG is P_s2p_u/[1-(1-P_s2p_u)*(1-P_smove)], thus the defualt final probability is 0.2703
                         'P_p2s_u': 0,  # probability of transformation from 80S to 40S at uSTOP in a single action (i.e., reinitiation)
                         'P_s2p_m': 0.9,
                         # probability of transformation from 40S to 80S at mATG in a single action
                         # CAUTION: in the article, the symbol of this parameter is I_CDS.
                         # final transforming probability at mATG is P_s2p_m/[1-(1-P_s2p_m)*(1-P_smove)], thus the defualt final probability is 0.9677
                         'P_p2s_m': 0,  # probability of transformation from 80S to 40S at uSTOP in a single action (i.e., reinitiation)
                         'P_smove': 0.3,
                         # probability that a 40S moves to the next position in a single action
                         # CAUTION: in the article, the symbol of this parameter is v_s.
                         'P_pmove_u': 0.3,
                         # probability that a 80S located in uORF moves to the next position in a single action
                         # CAUTION: in the article, the symbol of this parameter is v_Eu.
                         'P_pmove_m': 0.5,
                         # probability that a 80S loacted in mORF moves to the next position in a single action
                         # CAUTION: in the article, the symbol of this parameter is v_EC.
                         'P_sdeath': 0,  # probability of spontaneous 40S dissociation. ONLY happens when a 40S moves
                         'P_edeath': 0,  # probability of spontaneous 80S dissociation. ONLY happens when a 80S moves
                         'P_clear_up': 0,
                         # capacity of the 80S to remove the 40S in a 40S->80S collision, 0 means unable and 1 means able
                         # CAUTION: in the article, the symbol of this parameter is K_up.
                         'P_clear_down': 1,
                         # capacity of the 80S to remove the 40S in a 80S->40S collision, 0 means unable and 1 means able
                         # CAUTION: in the article, the symbol of this parameter is K_down.
                         'TTime': 1000000  # total number of actions in a single run of simulation
                         }


@jit(nopython=True)
def Free2go(Path, jstart, Len_rib):
    return sum(Path[jstart + 1:jstart + Len_rib + 1])


@jit(nopython=True)
def Free2transform(Path, jstart, Len_rib):
    return sum(Path[jstart + 1:jstart + Len_rib])


@jit(nopython=True)
def Want2Jump(j, Path, Len_a, Len_u, Ls, Le, P_smove, P_pmove_u, P_pmove_m, P_clear_up, P_clear_down):
    flag = False
    if Path[j] == 1 and Path[j + Ls] != 1:
        P_smove_final = P_smove if Path[j + Ls] == 0 else P_smove * P_clear_up
        if random.random() < P_smove_final:
            flag = True
    elif Path[j] == 2 and Path[j + Le] < 2:
        if j < Len_a + Len_u:
            P_pmove_u_final = P_pmove_u if Path[j + Le] == 0 else P_pmove_u * P_clear_down
            if random.random() < P_pmove_u_final:
                flag = True
        elif j >= Len_a + Len_u:
            P_pmove_m_final = P_pmove_m if Path[j + Le] == 0 else P_pmove_m * P_clear_down
            if random.random() < P_pmove_m_final:
                flag = True
    return flag


@jit(nopython=True)
def MutateS2E(P_s2p, P_delay_a, flag, freeflag):
    rib = flag
    if flag == 1 and random.random() < P_s2p:
        rib = 3
    elif flag == 3 and random.random() < P_delay_a and freeflag == 0:
        rib = 2
    return rib


@jit(nopython=True)
def MutateE2S(P_p2s, P_delay_b, flag, freeflag):
    rib = flag
    if flag == 2:
        rib = 4
    elif flag == 4 and random.random() < P_delay_b and freeflag == 0:
        if random.random() < P_p2s:
            rib = 1
        else:
            rib = 0
    return rib


def Sim_main(condition_set=None):
    if condition_set is None:
        condition_set = default_condition_set
    Ls = condition_set['Ls']
    Le = condition_set['Le']
    Len_a = condition_set['Len_a']
    Len_u = condition_set['Len_u']
    Len_b = condition_set['Len_b']
    Len_m = condition_set['Len_m']
    Len_c = condition_set['Len_c']
    Lambda = condition_set['Lambda']
    P_delay_au = condition_set['P_delay_au']
    P_delay_ub = condition_set['P_delay_ub']
    P_delay_bm = condition_set['P_delay_bm']
    P_delay_mc = condition_set['P_delay_mc']
    P_s2p_u = condition_set['P_s2p_u']
    P_p2s_u = condition_set['P_p2s_u']
    P_s2p_m = condition_set['P_s2p_m']
    P_p2s_m = condition_set['P_p2s_m']
    P_smove = condition_set['P_smove']
    P_pmove_u = condition_set['P_pmove_u']
    P_pmove_m = condition_set['P_pmove_m']
    P_sdeath = condition_set['P_sdeath']
    P_edeath = condition_set['P_edeath']
    P_clear_up = condition_set['P_clear_up']
    P_clear_down = condition_set['P_clear_down']
    TTime = condition_set['TTime']
    res = Sim_single_set(Ls, Le, Len_a, Len_u, Len_b, Len_m, Len_c, Lambda, P_delay_au, P_delay_ub, P_delay_bm,
                         P_delay_mc, P_s2p_u, P_p2s_u, P_s2p_m, P_p2s_m, P_smove, P_pmove_u, P_pmove_m, P_sdeath,
                         P_edeath, P_clear_up, P_clear_down, TTime)
    (SRNA_started, SRNA_passed_au, PRNA_started_u, PRNA_dropped_u, PRNA_finished_u, SRNA_passed_bm, PRNA_started_m,
     PRNA_dropped_m, PRNA_finished_m, SRNA_dropped, SRNA_end, SRNA_kicked_upstream_u, SRNA_kicked_downstream_u,
     SRNA_kicked_upstream_m, SRNA_kicked_downstream_m) = res
    sim_ID = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return {'sim_ID': sim_ID,
            'condition_set': condition_set, 'SRNA_started': SRNA_started,
            'SRNA_passed_au': SRNA_passed_au, 'PRNA_started_u': PRNA_started_u, 'PRNA_dropped_u': PRNA_dropped_u,
            'PRNA_finished_u': PRNA_finished_u,
            'SRNA_passed_bm': SRNA_passed_bm, 'PRNA_started_m': PRNA_started_m, 'PRNA_dropped_m': PRNA_dropped_m,
            'PRNA_finished_m': PRNA_finished_m,
            'SRNA_dropped': SRNA_dropped, 'SRNA_end': SRNA_end,
            'SRNA_kicked_upstream_u': SRNA_kicked_upstream_u, 'SRNA_kicked_downstream_u': SRNA_kicked_downstream_u,
            'SRNA_kicked_upstream_m': SRNA_kicked_upstream_m, 'SRNA_kicked_downstream_m': SRNA_kicked_downstream_m}


@jit(nopython=True)
def Sim_single_set(Ls, Le, Len_a, Len_u, Len_b, Len_m, Len_c, Lambda, P_delay_au, P_delay_ub, P_delay_bm, P_delay_mc,
                   P_s2p_u, P_p2s_u, P_s2p_m, P_p2s_m, P_smove, P_pmove_u, P_pmove_m, P_sdeath, P_edeath, P_clear_up,
                   P_clear_down, TTime):
    Len_tot = Len_a + Len_u + Len_b + Len_m + Len_c

    # Path = np.zeros(Len_tot, dtype=int)
    Path = [0] * Len_tot
    SRNA_started = 0
    SRNA_passed_au = 0
    PRNA_started_u = 0
    PRNA_finished_u = 0
    SRNA_passed_bm = 0
    PRNA_started_m = 0
    PRNA_finished_m = 0
    SRNA_dropped = 0
    PRNA_dropped_u = 0
    PRNA_dropped_m = 0
    SRNA_end = 0
    SRNA_kicked_upstream_u = 0
    SRNA_kicked_downstream_u = 0
    SRNA_kicked_upstream_m = 0
    SRNA_kicked_downstream_m = 0

    for i in range(0, TTime):

        if Free2go(Path, -1, Ls) == 0 and random.random() < Lambda:  # try loading new ribosome
            Path[0] = 1
            SRNA_started = SRNA_started + 1

        if Len_u > 1:  # try assembly at uATG and dissociate at uSTOP
            fl_au = Free2transform(Path, Len_a, Le)
            uATG_record = Path[Len_a]
            Path[Len_a] = MutateS2E(P_s2p_u, P_delay_au, Path[Len_a], fl_au)
            if uATG_record != 3 and Path[Len_a] == 3:
                PRNA_started_u = PRNA_started_u + 1
            fl_ub = Free2transform(Path, Len_a + Len_u - 1, Ls)
            uSTOP_record = Path[Len_a + Len_u - 1]
            Path[Len_a + Len_u - 1] = MutateE2S(P_p2s_u, P_delay_ub, Path[Len_a + Len_u - 1], fl_ub)
            if uSTOP_record != 4 and Path[Len_a + Len_u - 1] == 4:
                PRNA_finished_u = PRNA_finished_u + 1

        fl_bm = Free2transform(Path, Len_a + Len_u + Len_b, Le)
        mATG_record = Path[Len_a + Len_u + Len_b]
        Path[Len_a + Len_u + Len_b] = MutateS2E(P_s2p_m, P_delay_bm, Path[Len_a + Len_u + Len_b], fl_bm)
        if mATG_record != 3 and Path[Len_a + Len_u + Len_b] == 3:
            PRNA_started_m = PRNA_started_m + 1
        fl_mc = Free2transform(Path, Len_a + Len_u + Len_b + Len_m - 1, Ls)
        mSTOP_record = Path[Len_a + Len_u + Len_b + Len_m - 1]
        Path[Len_a + Len_u + Len_b + Len_m - 1] = MutateE2S(P_p2s_m, P_delay_mc,
                                                            Path[Len_a + Len_u + Len_b + Len_m - 1], fl_mc)
        if mSTOP_record != 4 and Path[Len_a + Len_u + Len_b + Len_m - 1] == 4:
            PRNA_finished_m = PRNA_finished_m + 1

        if Path[Len_tot - Ls] > 0 and random.random() < P_smove:
            Path[Len_tot - Ls] = 0
            SRNA_end = SRNA_end + 1

        Path_new = Path.copy()

        for j in range(0, Len_tot - Ls):  # 40S ribosomes moving
            if Path[j] == 1 and Want2Jump(j, Path, Len_a, Len_u, Ls, Le, P_smove, P_pmove_u, P_pmove_m, P_clear_up,
                                          P_clear_down):
                Path_new[j] = 0
                Path_new[j + 1] = 1
                if random.random() < P_sdeath:
                    SRNA_dropped = SRNA_dropped + 1
                    Path_new[j + 1] = 0

        for j in range(0, Len_tot - Ls):  # 80S ribosomes moving and clearing 40S they run into
            if Path[j] >= 2:
                if Path[j] == 2 and Want2Jump(j, Path, Len_a, Len_u, Ls, Le, P_smove, P_pmove_u, P_pmove_m, P_clear_up,
                                              P_clear_down):
                    Path_new[j] = 0
                    Path_new[j + 1] = 2
                    if random.random() < P_edeath:
                        if j < Len_a + Len_u:
                            PRNA_dropped_u = PRNA_dropped_u + 1
                        else:
                            PRNA_dropped_m = PRNA_dropped_m + 1
                        Path_new[j + 1] = 0
                    elif Path_new[j + Le] == 1:  # 80S ribosome clearing its way
                        if j + 1 < Len_a + Len_u:
                            SRNA_kicked_downstream_u = SRNA_kicked_downstream_u + 1
                        else:
                            SRNA_kicked_downstream_m = SRNA_kicked_downstream_m + 1
                        Path_new[j + Le] = 0
                elif Path_new[j - Ls + 1] == 1:  # 40S running into 80S and crashing
                    if j < Len_a + Len_u:
                        SRNA_kicked_upstream_u = SRNA_kicked_upstream_u + 1
                    else:
                        SRNA_kicked_upstream_m = SRNA_kicked_upstream_m + 1
                    Path_new[j - Ls + 1] = 0

        if Path[Len_a - 1] > 0 and Path_new[Len_a - 1] == 0:
            SRNA_passed_au = SRNA_passed_au + 1

        if Path[Len_a + Len_u + Len_b - 1] > 0 and Path_new[Len_a + Len_u + Len_b - 1] == 0:
            SRNA_passed_bm = SRNA_passed_bm + 1

        Path = Path_new

    return (
        SRNA_started, SRNA_passed_au, PRNA_started_u, PRNA_dropped_u, PRNA_finished_u, SRNA_passed_bm, PRNA_started_m,
        PRNA_dropped_m, PRNA_finished_m, SRNA_dropped, SRNA_end, SRNA_kicked_upstream_u, SRNA_kicked_downstream_u,
        SRNA_kicked_upstream_m, SRNA_kicked_downstream_m)


class Sim_main_mp:
    def __init__(self, condition_set_items, sim_res_itmes):
        self.condition_set_items = condition_set_items
        self.sim_res_itmes = sim_res_itmes

    def __call__(self, condition_set):
        Sim_res = Sim_main(condition_set)
        res_str = Sim_res['sim_ID'] + '\t' + '\t'.join(
            str(Sim_res['condition_set'][p]) for p in self.condition_set_items) + '\t' + '\t'.join(
            str(Sim_res[p]) for p in self.sim_res_itmes)
        print(res_str, flush=True)
        return Sim_res


def Sim_multi_condition(condition_list, outfile_path):
    condition_set_items = ['Ls', 'Le', 'Len_a', 'Len_u', 'Len_b', 'Len_m', 'Len_c', 'Lambda', 'P_delay_au',
                           'P_delay_ub',
                           'P_delay_bm', 'P_delay_mc', 'P_s2p_u', 'P_p2s_u', 'P_s2p_m', 'P_p2s_m', 'P_smove',
                           'P_pmove_u',
                           'P_pmove_m', 'P_sdeath', 'P_edeath', 'P_clear_up', 'P_clear_down', 'TTime']
    sim_res_itmes = ['SRNA_started', 'SRNA_passed_au', 'PRNA_started_u', 'PRNA_dropped_u', 'PRNA_finished_u',
                     'SRNA_passed_bm', 'PRNA_started_m', 'PRNA_dropped_m', 'PRNA_finished_m', 'SRNA_dropped',
                     'SRNA_end', 'SRNA_kicked_upstream_u', 'SRNA_kicked_downstream_u',
                     'SRNA_kicked_upstream_m', 'SRNA_kicked_downstream_m']
    outfile = open(outfile_path, 'w')
    header = 'sim_ID' + '\t' + '\t'.join(condition_set_items) + '\t' + '\t'.join(sim_res_itmes) + '\n'
    outfile.write(header)
    print(header, end='', flush=True)
    Sim_main_mp_func = Sim_main_mp(condition_set_items, sim_res_itmes)
    with multiprocessing.Pool(int(0.75 * multiprocessing.cpu_count())) as mp:
        res = mp.map(Sim_main_mp_func, condition_list)
    for j in res:
        res_str = j['sim_ID'] + '\t' + '\t'.join(
            str(j['condition_set'][p]) for p in condition_set_items) + '\t' + '\t'.join(
            str(j[p]) for p in sim_res_itmes) + '\n'
        outfile.write(res_str)
    outfile.close()
