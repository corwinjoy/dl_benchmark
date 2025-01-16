# Python
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow.parquet.encryption as pe
import pyarrow.dataset as ds
from deltalake.writer import write_deltalake
from deltalake import DeltaTable
import deltalake as dl
import shutil
import os
import time
import math
import base64

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

# Use the deltalake table reader
def read_delta_table(dl_path):
    dt = DeltaTable(dl_path)
    df_files = dt.file_uris()
    df_dl = dt.to_pandas()
    return (df_files, df_dl)

# Use pyarrow to directly read the underlying parquet files and merge
def read_delta_files(df_files):
    tbls = []
    for file in df_files:
        tbl = pq.read_table(file)
        tbls.append(tbl.to_pandas())
    all_tbls = pd.concat(tbls, ignore_index=True)
    return all_tbls

def run_bm_scenarios():
    # One gigabyte of float32 data translates to the following number of rows and cols
    # 1024^3 (1 GiB) / 4 (bytes per float 32) / 128 cols
    # 2_097_152
    df_bm = None
    nrows = math.ceil(2_097_152 / 10) # Up to 10 reps
    for nappend in range(0, 10):
        for ncols in range(20, 140, 20):
            clear_folder(dl_folder)
            df = gen_df(nrows, ncols)
            write_deltalake(dl_path, df)
            if nappend > 0:
                for i in range(nappend):
                    df = gen_df(nrows, ncols)
                    write_deltalake(dl_path, df, mode="append")

            tbl_tm = timed(read_delta_table, dl_path)
            (df_files, df_dl) = tbl_tm[0]
            tm_delta = tbl_tm[1]
            # print("delta table:")
            # print(df_dl)
            pq_tm = timed(read_delta_files, df_files)
            tm_pq = pq_tm[1]
            # print("pq table:")
            # print(pq_tm[0])
            df_tmp = pd.DataFrame({"nrows": [nrows], "ncols": [ncols], "nappend": [nappend],
                                   "time_delta": [tm_delta], "time_pq": [tm_pq]})
            print(df_tmp)
            if df_bm is None:
                df_bm = df_tmp
            else:
                df_bm = pd.concat([df_bm, df_tmp], ignore_index=True)

    print("\n----------------------------------------------\n")
    print("Collected benchmark results:")
    print(df_bm)
    df_bm.to_csv(dl_folder + "/../delta_benchmark.csv")


FOOTER_KEY = b"0123456789112345"
FOOTER_KEY_NAME = "footer_key"
COL_KEY = b"1234567890123450"
COL_KEY_NAME = "col_key"

def basic_encryption_config(df):
    basic_encryption_config = pe.EncryptionConfiguration(
        footer_key=FOOTER_KEY_NAME,
        column_keys={
            COL_KEY_NAME: df.columns,
        })
    return basic_encryption_config


class InMemoryKmsClient(pe.KmsClient):
    """This is a mock class implementation of KmsClient, built for testing
    only.
    """

    def __init__(self, config):
        """Create an InMemoryKmsClient instance."""
        pe.KmsClient.__init__(self)
        self.master_keys_map = config.custom_kms_conf

    def wrap_key(self, key_bytes, master_key_identifier):
        """Not a secure cipher - the wrapped key
        is just the master key concatenated with key bytes"""
        master_key_bytes = self.master_keys_map[master_key_identifier].encode(
            'utf-8')
        wrapped_key = b"".join([master_key_bytes, key_bytes])
        result = base64.b64encode(wrapped_key)
        return result

    def unwrap_key(self, wrapped_key, master_key_identifier):
        """Not a secure cipher - just extract the key from
        the wrapped key"""
        expected_master_key = self.master_keys_map[master_key_identifier]
        decoded_wrapped_key = base64.b64decode(wrapped_key)
        master_key_bytes = decoded_wrapped_key[:16]
        decrypted_key = decoded_wrapped_key[16:]
        if (expected_master_key == master_key_bytes.decode('utf-8')):
            return decrypted_key
        raise ValueError("Incorrect master key used",
                         master_key_bytes, decrypted_key)

def run_encrypt_scenarios():
    nrows = math.ceil(2_097_152 / 10)  # Up to 10 reps
    nappend = 1
    ncols = 20
    clear_folder(dl_folder)
    df = gen_df(nrows, ncols)
    encryption_config = basic_encryption_config(df)
    kms_connection_config = pe.KmsConnectionConfig(
        custom_kms_conf={
            FOOTER_KEY_NAME: FOOTER_KEY.decode("UTF-8"),
            COL_KEY_NAME: COL_KEY.decode("UTF-8"),
        }
    )

    def kms_factory(kms_connection_configuration):
        return InMemoryKmsClient(kms_connection_configuration)

    crypto_factory = pe.CryptoFactory(kms_factory)
    file_encryption_properties = crypto_factory.file_encryption_properties(
        kms_connection_config, encryption_config)
    assert file_encryption_properties is not None

    parquet_format = ds.ParquetFileFormat()
    write_options = parquet_format.make_write_options(encryption_properties=file_encryption_properties)
    write_deltalake(dl_path, df, file_options=write_options)
    if nappend > 0:
        for i in range(nappend):
            df = gen_df(nrows, ncols)
            write_deltalake(dl_path, df, mode="append", file_options=write_options)

    tbl_tm = timed(read_delta_table, dl_path)
    (df_files, df_dl) = tbl_tm[0]
    tm_delta = tbl_tm[1]
    print("delta table:")
    print(df_dl)


run_encrypt_scenarios()