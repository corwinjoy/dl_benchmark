from common import *
import pyarrow as pa
import pyarrow.parquet.encryption as pe
import pyarrow.dataset as ds
from deltalake.writer import write_deltalake
from deltalake import DeltaTable
import math
import base64

FOOTER_KEY = b"0123456789112345"
FOOTER_KEY_NAME = "footer_key"
COL_KEY = b"1234567890123450"
COL_KEY_NAME = "col_key"

def read_delta_table(dl_path):
    dt = DeltaTable(dl_path)
    df_files = dt.file_uris()
    df_dl = dt.to_pandas()
    return (df_files, df_dl)

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

def create_encryption_config(df):
    return pe.EncryptionConfiguration(
        footer_key=FOOTER_KEY_NAME,
        column_keys={
            COL_KEY_NAME: df.columns.tolist(),
        })


def create_decryption_config():
    return pe.DecryptionConfiguration(cache_lifetime=300)


def create_kms_connection_config():
    return pe.KmsConnectionConfig(
        custom_kms_conf={
            FOOTER_KEY_NAME: FOOTER_KEY.decode("UTF-8"),
            COL_KEY_NAME: COL_KEY.decode("UTF-8"),
        }
    )

def kms_factory(kms_connection_configuration):
    return InMemoryKmsClient(kms_connection_configuration)


def run_delta_ecrypt():
    # Test encryption with delta lake
    # The interface looks like it should support encryption, but
    # does not support encryption in practice.
    nrows = math.ceil(2_097_152 / 10)  # Up to 10 reps
    nappend = 1
    ncols = 20
    clear_folder(dl_folder)
    df = gen_df(nrows, ncols)

    encryption_config = create_encryption_config(df)
    decryption_config = create_decryption_config()
    kms_connection_config = create_kms_connection_config()

    crypto_factory = pe.CryptoFactory(kms_factory)
    parquet_encryption_cfg = ds.ParquetEncryptionConfig(
        crypto_factory, kms_connection_config, encryption_config
    )
    parquet_decryption_cfg = ds.ParquetDecryptionConfig(
        crypto_factory, kms_connection_config, decryption_config
    )
    pq_scan_opts = ds.ParquetFragmentScanOptions(
        decryption_config=parquet_decryption_cfg
    )

    parquet_format = ds.ParquetFileFormat()
    write_options = parquet_format.make_write_options(encryption_config=parquet_encryption_cfg)

    # It seems the engine is not fully setup for encryption.
    # Anyway, the pyarrow engine is deprecated
    write_deltalake(dl_path, df, file_options=write_options, engine = "pyarrow")
    if nappend > 0:
        for i in range(nappend):
            df = gen_df(nrows, ncols)
            write_deltalake(dl_path, df, mode="append", file_options=write_options)

    tbl_tm = timed(read_delta_table, dl_path)
    (df_files, df_dl) = tbl_tm[0]
    tm_delta = tbl_tm[1]
    print("delta table:")
    print(df_dl)


def run_pq_encrypt():
    # Test encryption with pyarrow
    nrows = math.ceil(2_097_152 / 10)  # Up to 10 reps
    ncols = 128 # 100 MB of data
    clear_folder(pq_folder)
    df = gen_df(nrows, ncols)
    table = pa.Table.from_pandas(df)

    encryption_config = create_encryption_config(df)
    decryption_config = create_decryption_config()
    kms_connection_config = create_kms_connection_config()

    crypto_factory = pe.CryptoFactory(kms_factory)
    parquet_encryption_cfg = ds.ParquetEncryptionConfig(
        crypto_factory, kms_connection_config, encryption_config
    )
    parquet_decryption_cfg = ds.ParquetDecryptionConfig(
        crypto_factory, kms_connection_config, decryption_config
    )

    def read_encrypted_data(parquet_decryption_cfg, df):
        # set decryption config for parquet fragment scan options
        pq_scan_opts = ds.ParquetFragmentScanOptions(
            decryption_config=parquet_decryption_cfg
        )
        pformat = ds.ParquetFileFormat(default_fragment_scan_options=pq_scan_opts)
        dataset = ds.dataset(pq_folder, format=pformat)
        tbl_read = dataset.to_table()
        df2 = tbl_read.to_pandas()
        # pd.testing.assert_frame_equal(df, df2, rtol=1e-2, atol=1e-2)
        return df2

    def read_unencrypted_data(df):
        # set decryption config for parquet fragment scan options
        dataset = ds.dataset(pq_folder, format="parquet")
        tbl_read = dataset.to_table()
        df2 = tbl_read.to_pandas()
        # pd.testing.assert_frame_equal(df, df2, rtol=1e-2, atol=1e-2)
        return df2

    # create write_options with dataset encryption config
    pformat = ds.ParquetFileFormat()
    write_options = pformat.make_write_options(encryption_config=parquet_encryption_cfg)

    ds.write_dataset(
        data=table,
        base_dir=pq_folder,
        format=pformat,
        file_options=write_options,
        use_threads=False # Maintain original row order
    )
    decrypt_res = timed(read_encrypted_data, parquet_decryption_cfg, df)
    decrypt_tm = decrypt_res[1]

    clear_folder(pq_folder)
    ds.write_dataset(
        data=table,
        base_dir=pq_folder,
        format="parquet",
        use_threads=False # Maintain original row order
    )
    regular_res = timed(read_unencrypted_data, df)
    regular_tm = regular_res[1]

    print(f"Unencrypted read time: {regular_tm}")
    print(f"Encrypted read time: {decrypt_tm}")
    pct = (decrypt_tm/regular_tm)*100.0
    print(f"Encrypted/Unencrypted %: {pct}")

# Run deltalake benchmarks
# run_bm_scenarios()

# Benchmark encryption of parquet
# run_pq_encrypt()

# Try deltalake encryption
run_delta_ecrypt()