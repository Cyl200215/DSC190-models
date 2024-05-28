import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
from glob import glob
import pickle
import gc
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import roc_auc_score 
from typing import Any
import json
import warnings
import base64
warnings.simplefilter(action='ignore', category=FutureWarning)


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        if str(col_type)=="category":
            continue
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            continue
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


class Pipeline:
    @staticmethod
    def set_table_dtypes(df):
        for col in df.columns:
            if col in ["case_id", "WEEK_NUM", "num_group1", "num_group2"]:
                df = df.with_columns(pl.col(col).cast(pl.Int64))
            elif col in ["date_decision"]:
                df = df.with_columns(pl.col(col).cast(pl.Date))
            elif col[-1] in ("P", "A"):
                df = df.with_columns(pl.col(col).cast(pl.Float64))
            elif col[-1] in ("M",):
                df = df.with_columns(pl.col(col).cast(pl.String))
            elif col[-1] in ("D",):
                df = df.with_columns(pl.col(col).cast(pl.Date))
        return df

    @staticmethod
    def handle_dates(df):
        for col in df.columns:
            if col[-1] in ("D",):
                df = df.with_columns(pl.col(col) - pl.col("date_decision"))
                df = df.with_columns(pl.col(col).dt.total_days())
        df = df.drop("date_decision", "MONTH")

        return df

    @staticmethod
    def filter_cols(df):
        for col in df.columns:
            if col not in ["target", "case_id", "WEEK_NUM"]:
                isnull = df[col].is_null().mean()

                if isnull > 0.7:
                    df = df.drop(col)

        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()

                if (freq == 1) | (freq > 200):
                    df = df.drop(col)

        return df
    

class Aggregator:
    def num_expr(df):
        cols = [col for col in df.columns if col[-1] in ("P", "A")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]

        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        # expr_first = [pl.first(col).alias(f"first_{col}") for col in cols]
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        expr_median = [pl.median(col).alias(f"median_{col}") for col in cols]
        expr_var = [pl.var(col).alias(f"var_{col}") for col in cols]

        return expr_max + expr_last + expr_mean 

    def date_expr(df):
        cols = [col for col in df.columns if col[-1] in ("D")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        expr_mean = [pl.mean(col).alias(f"mean_{col}") for col in cols]
        expr_median = [pl.median(col).alias(f"median_{col}") for col in cols]

        return expr_max + expr_last + expr_mean 

    def str_expr(df):
        cols = [col for col in df.columns if col[-1] in ("M",)]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        return expr_max + expr_last

    def other_expr(df):
        cols = [col for col in df.columns if col[-1] in ("T", "L")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        return expr_max + expr_last

    def count_expr(df):
        cols = [col for col in df.columns if "num_group" in col]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        expr_last = [pl.last(col).alias(f"last_{col}") for col in cols]
        return expr_max + expr_last

    def get_exprs(df):
        exprs = Aggregator.num_expr(df) + \
                Aggregator.date_expr(df) + \
                Aggregator.str_expr(df) + \
                Aggregator.other_expr(df) + \
                Aggregator.count_expr(df)
        return exprs
    
class VotingModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        y_preds = [estimator.predict(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

    def predict_proba(self, X):
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

def predict(dataPath, pkl_path):

    with open(pkl_path, 'r') as f:
        loaded_data = json.load(f)


    cat_cols = loaded_data['cat_col']
    df_train_cols = loaded_data['df_train_col']
    uses = loaded_data['uses']

    def read_file(path, depth=None):
        df = pl.read_parquet(path)
        df = df.pipe(Pipeline.set_table_dtypes)
        if depth in [1, 2]:
            df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
        return df

    def read_files(regex_path, depth=None):
        chunks = []
        for path in glob(str(regex_path)):
            chunks.append(pl.read_parquet(path).pipe(Pipeline.set_table_dtypes))
        df = pl.concat(chunks, how="vertical_relaxed")
        if depth in [1, 2]:
            df = df.group_by("case_id").agg(Aggregator.get_exprs(df))
            
        del chunks
        gc.collect()
            
        return df
    

    def feature_eng(df_base, depth_0, depth_1, depth_2):
        df_base = (
            df_base
            .with_columns(
                month_decision = pl.col("date_decision").dt.month(),
                weekday_decision = pl.col("date_decision").dt.weekday(),
            )
        )

        for i, df in enumerate(depth_0 + depth_1 + depth_2):
            df_base = df_base.join(df, how="left", on="case_id", suffix=f"_{i}")
        return df_base
    
    def to_pandas(df_data, cat_cols=None):
        df_data = df_data.to_pandas()
        if cat_cols is None:
            cat_cols = list(df_data.select_dtypes("object").columns)
        df_data[cat_cols] = df_data[cat_cols].astype("category")
        return df_data, cat_cols
    

    test_credit_bureau_a_1 = read_files(dataPath + "/test_credit_bureau_a_1_*.parquet", 1)
    test_credit_bureau_a_1 = test_credit_bureau_a_1.with_columns(
        ((pl.col('max_dateofcredend_289D') - pl.col('max_dateofcredstart_739D')).dt.total_days()).alias('max_credit_duration_daysA')
    ).with_columns(
        ((pl.col('max_dateofcredend_353D') - pl.col('max_dateofcredstart_181D')).dt.total_days()).alias('max_closed_credit_duration_daysA')
    ).with_columns(
        ((pl.col('max_dateofrealrepmt_138D') - pl.col('max_overdueamountmax2date_1002D')).dt.total_days()).alias('max_time_from_overdue_to_closed_realrepmtA')
    ).with_columns(
        ((pl.col('max_dateofrealrepmt_138D') - pl.col('max_overdueamountmax2date_1142D')).dt.total_days()).alias('max_time_from_active_overdue_to_realrepmtA')
    )

    test_credit_bureau_b_1 = read_file(dataPath + "/test_credit_bureau_b_1.parquet", 1)
    test_credit_bureau_b_1 = test_credit_bureau_b_1.with_columns(
        ((pl.col('max_contractmaturitydate_151D') - pl.col('max_contractdate_551D')).dt.total_days()).alias('contract_duration_days_A')
    ).with_columns(
        ((pl.col('max_lastupdate_260D') - pl.col('max_contractdate_551D')).dt.total_days()).alias('last_update_duration_days_A')
    )

    test_static = read_files(dataPath + "/test_static_0_*.parquet")
    condition_all_nan = (
        pl.col('maxdbddpdlast1m_3658939P').is_null() &
        pl.col('maxdbddpdtollast12m_3658940P').is_null() &
        pl.col('maxdbddpdtollast6m_4187119P').is_null()
    )

    condition_exceed_thresholds = (
        (pl.col('maxdbddpdlast1m_3658939P') > 31) |
        (pl.col('maxdbddpdtollast12m_3658940P') > 366) |
        (pl.col('maxdbddpdtollast6m_4187119P') > 184)
    )

    test_static = test_static.with_columns(
        pl.when(condition_all_nan | condition_exceed_thresholds)
        .then(0)
        .otherwise(1)
        .alias('max_dbddpd_boolean_P')
    )

    test_static = test_static.with_columns(
        pl.when(
            (pl.col('maxdbddpdlast1m_3658939P') <= 0) &
            (pl.col('maxdbddpdtollast12m_3658940P') <= 0) &
            (pl.col('maxdbddpdtollast6m_4187119P') <= 0)
        )
        .then(1)
        .otherwise(0)
        .alias('max_pays_debt_on_time_P')
    )

    test_static = test_static.with_columns(
        pl.when((pl.col('firstdatedue_489D') <= pl.col('datefirstoffer_1144D'))).then(1).otherwise(0).alias('firstdatedue_before_offer_P')
    ).with_columns(
        pl.when((pl.col('datelastunpaid_3546854D').is_null())).then(1).otherwise(0).alias('missed_payment_P')
    ).with_columns(
        pl.when((pl.col('maxdpdinstldate_3546855D').is_null())).then(1).otherwise(0).alias('late_payment_P')
    ).with_columns(
        pl.when((pl.col('lastdelinqdate_224D').is_null())).then(1).otherwise(0).alias('any_delinquency_P')
    ).with_columns(
        ((pl.col('dtlastpmtallstes_4499206D') - pl.col('datelastunpaid_3546854D')).dt.total_days()).alias('days_between_last_unpaid_and_last_payment_A')
    ).with_columns(
        ((pl.col('dtlastpmtallstes_4499206D') - pl.col('lastdelinqdate_224D')).dt.total_days()).alias('days_between_last_default_and_last_payment_A')
    ).with_columns(
        ((pl.col('datelastinstal40dpd_247D') - pl.col('firstdatedue_489D')).dt.total_days()).alias('days_between_first_and_last_installment_A')
    ).with_columns(
        ((pl.col('maxdpdinstldate_3546855D') - pl.col('lastdelinqdate_224D')).dt.total_days()).alias('days_between_max_dpd_and_last_default_A')
    ).with_columns(
        ((pl.col('maxdpdinstldate_3546855D') - pl.col('dtlastpmtallstes_4499206D')).dt.total_days()).alias('days_between_max_dpd_and_last_payment_A')
    )

    # read test files
    data_store = {
        "df_base": read_file(dataPath + "/test_base.parquet"),
        "depth_0": [
            read_file(dataPath + "/test_static_cb_0.parquet"),
            test_static,
        ],
        "depth_1": [
            read_files(dataPath + "/test_applprev_1_*.parquet", 1),
            read_file(dataPath + "/test_tax_registry_a_1.parquet", 1),
            read_file(dataPath + "/test_tax_registry_b_1.parquet", 1),
            read_file(dataPath + "/test_tax_registry_c_1.parquet", 1),
            test_credit_bureau_a_1,
            test_credit_bureau_b_1,
            read_file(dataPath + "/test_other_1.parquet", 1),
            read_file(dataPath + "/test_person_1.parquet", 1),
            read_file(dataPath + "/test_deposit_1.parquet", 1),
            read_file(dataPath + "/test_debitcard_1.parquet", 1),
        ],
        "depth_2": [
            read_file(dataPath + "/test_credit_bureau_b_2.parquet", 2),
        ]
    }

    df_test = feature_eng(**data_store)
    df_test = df_test.pipe(Pipeline.handle_dates)
    print("test data shape:\t", df_test.shape)
    del data_store
    gc.collect()

    df_test = df_test.with_columns(
        pl.when((pl.col('dtlastpmtallstes_4499206D') <= 365)).then(1).otherwise(0).alias('pay_among_year_P')
    ).with_columns(
        pl.when((pl.col('dtlastpmtallstes_4499206D') <= 90)).then(1).otherwise(0).alias('pay_among_three_months_P')
    )

    df_test = df_test.select([col for col in df_train_cols if col != "target"])
    df_test, cat_cols = to_pandas(df_test, cat_cols)
    df_test = reduce_mem_usage(df_test)
    df_test = df_test[[col for col in uses if col != "target"]].drop(columns = ['WEEK_NUM'])
    case_ids = df_test["case_id"]
    df_test = df_test.set_index('case_id')

    model = pickle.loads(base64.b64decode(loaded_data['model'].encode('utf-8')))
    score = model.predict_proba(df_test)[:, 1]
    
    df_submission = pd.DataFrame({
        "case_id": case_ids,
        "score": score
    }).set_index('case_id')

    return df_submission

data_path = "../../../data/home-credit-credit-risk-model-stability/parquet_files/test"
pkl_path = "data.json"
df_submission = predict(data_path, pkl_path)