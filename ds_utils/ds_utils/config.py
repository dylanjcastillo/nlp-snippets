import pandas as pd


def set_display_options():
    # Source: https://towardsdatascience.com/10-python-skills-419e5e4c4d66
    pd.set_option("max_colwidth", 1000)
    pd.set_option("max_rows", 50)
    pd.set_option("max_columns", 100)
    pd.options.display.float_format = "{:,.2f}".format
