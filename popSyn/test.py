from random import seed
import pandas as pd
import numpy as np
from pyBN.learning.structure.score.tabu import tabu
from pyBN.learning.structure.score.hill_climbing import hc
from pyBN.plotting.plot import plot_nx
from pyBN.inference.marginal_approx.gibbs_sample import gibbs_sample
from pyBN.inference.marginal_approx.forward_sample import forward_sample
from pyBN.learning.parameter.mle import mle_estimator
from pyBN.learning.parameter.bayes import bayes_estimator
from checker import SRMSE


if __name__ == "__main__":
    ATTRIBUTES = ['AGEGROUP', 'CARLICENCE', 'SEX', 'PERSINC', 'DWELLTYPE', 'TOTALVEHS']

    # import data
    p_original_df = pd.read_csv("../data/VISTA_2012_16_v1_SA1_CSV/P_VISTA12_16_SA1_V1.csv")
    # Only have record of the main person (the person that did the survey)
    p_self_df = p_original_df[p_original_df['RELATIONSHIP']=='Self']
    h_original_df = pd.read_csv("../data/VISTA_2012_16_v1_SA1_CSV/H_VISTA12_16_SA1_V1.csv")

    orignal_df = pd.merge(p_self_df, h_original_df, on=['HHID'])
    df = orignal_df[ATTRIBUTES].dropna()

    make_like_paper = True
    if make_like_paper:
        df.loc[df['TOTALVEHS'] == 0, 'TOTALVEHS'] = 'NO'
        df.loc[df['TOTALVEHS'] != 'NO', 'TOTALVEHS'] = 'YES'

        df.loc[df['CARLICENCE'] == 'No Car Licence', 'CARLICENCE'] = 'NO'
        df.loc[df['CARLICENCE'] != 'NO', 'CARLICENCE'] = 'YES'
    # print(df)

    # Convert to cardi to work with this lib
    df_work = df.copy()
    ref_dict = {}
    for att in ATTRIBUTES:
        ls_vals = list(pd.unique(df_work[att]))
        ref_dict[att] = ls_vals
        df_work[att].replace(ls_vals, range(len(ls_vals)), inplace=True)

    # take some random from population
    seed_df = df_work.sample(n = 17000)
    data = seed_df.to_numpy()

    # Struct learning
    bn = tabu(data, debug=False)
    # Para learning
    bayes_estimator(bn, data)
    # plot_nx(bn)

    # SAMPLING
    prob_dict, samples = forward_sample(bn, n=5)
    sample_df = pd.DataFrame(samples)
    print(sample_df)
    print(df_work)
    print('DONE')