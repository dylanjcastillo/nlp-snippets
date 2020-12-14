import os
import random

import numpy as np
import pandas as pd


def set_random_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


def set_display_options():
    pd.set_option("max_colwidth", 1000)
    pd.set_option("max_rows", 50)
    pd.set_option("max_columns", 100)
    pd.options.display.float_format = "{:,.2f}".format
