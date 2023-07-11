"""

This file contains utils functions for exploration of the setup functionality of PyCaret classification module.

"""

# Imports
from pandas import read_csv

# Define constants
EXPERIMENT = 'Edema'
FILE_DF = '../../../data/cxr_ic_fusion_1103.csv'
MODEL = 'xgboost'
N_DATA = 45050
OPTIMIZE = 'AUC'
PREDICTIVE_COLUMNS_PREFIX = ['de_', 'vd_', 'vp_', 'vmd_', 'vmp_', 'ts_ce_', 'ts_le_', 'ts_pe_', 'n_ecg_', 'n_ech_']
TUNING_GRID = {'max_depth': [5, 6, 7, 8],
               'n_estimators': [200, 300],
               'learning_rate': [0.3, 0.1, 0.05]}


def get_experiment_df():
    # Read data from local source
    df = read_csv(FILE_DF, nrows=N_DATA)

    # Get data where there is a value for EXPERIMENT
    df = df[df[EXPERIMENT].isin([0, 1])]

    # Keep columns for the prediction
    columns = ['haim_id', EXPERIMENT]
    for column_suffix in PREDICTIVE_COLUMNS_PREFIX:
        for df_column in df.columns:
            if df_column.startswith(column_suffix):
                columns.append(df_column)
    df = df[columns]
    return df
