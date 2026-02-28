# %%
import pandas as pd
import numpy as np
import sklearn as sk

# %%
salary=pd.read_csv("2025_salaries.csv", header=1, encoding="latin-1")
stats=pd.read_csv("nba_2025.csv", encoding="latin-1")

# %%
merged=pd.merge(salary, stats, on="Player")

# %%
dupes=merged[merged.duplicated(subset="Player", keep=False)]

# %%
