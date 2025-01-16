# Python
import pandas as pd
import numpy as np
from deltalake.writer import write_deltalake
from deltalake import DeltaTable
import shutil
import os
import time

def clear_folder(folder_path):
    """Clears the contents of a folder without deleting the folder itself."""

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


dl_folder = "/home/cjoy/src/dl_benchmark/tmp"
dl_path = dl_folder + "/deltars_table"

def gen_df(nrows, ncols):
    # Create a DataFrame with random data
    letter = 'x'
    cols = [f"{letter}{i}" for i in range(ncols)]
    df = pd.DataFrame(np.random.rand(nrows, ncols), columns=cols, dtype=pd.Float32Dtype())
    return df

def timed(func, *args, **kwargs):
    start = time.time()
    out = func(*args, **kwargs)
    return out, time.time() - start

def read_delta_table(dl_path):
    dt = DeltaTable(dl_path)
    df_files = dt.files()
    df_dl = dt.to_pandas()
    return (df_files, df_dl)

# One gigabyte of float32 data translates to the following number of rows and cols
# 1024^3 (1 GiB) / 4 (bytes per float 32) / 128 cols
# 2_097_152
df_bm = None
nrows = 2_097_152 / 10 # Up to 10 reps
for nappend in range(10):
    for ncols in range(20, 140, 20):
        clear_folder(dl_folder)
        df = gen_df(nrows, ncols)
        write_deltalake(dl_path, df)
        if nappend > 0:
            for i in range(nappend):
                df = gen_df(nrows, ncols)
                write_deltalake(dl_path, df, mode="append")

        tm_tbl = timed(read_delta_table, dl_path)
        tm = tm_tbl[1]
        df_tmp = pd.DataFrame({"nrows": [nrows], "ncols": [ncols], "nappend": [nappend], "time": [tm]})
        if df_bm is None:
            df_bm = df_tmp
        else:
            df_bm = pd.concat([df_bm, df_tmp], ignore_index=True)


print("Times to read table")
print(df_bm)
df_bm.to_csv(dl_folder + "/delta_benchmark.csv")


