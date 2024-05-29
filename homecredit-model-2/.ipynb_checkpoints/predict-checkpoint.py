import polars as pl
import numpy as np
import pandas as pd
import lightgbm as lgb
from glob import glob
import pickle
import gc
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Any, Union
import json
import warnings
import base64
warnings.simplefilter(action='ignore', category=FutureWarning)

dataPath = "../../../data/home-credit-credit-risk-model-stability/parquet_files/test"

class Utility:
    @staticmethod
    def get_feat_defs(ending_with: str) -> None:
        """
        Retrieves feature definitions from a CSV file based on the specified ending.

        Args:
        - ending_with (str): Ending to filter feature definitions.

        Returns:
        - pl.DataFrame: Filtered feature definitions.
        """
        feat_defs: pl.DataFrame = pl.read_csv(dataPath / "feature_definitions.csv")

        filtered_feats: pl.DataFrame = feat_defs.filter(
            pl.col("Variable").apply(lambda var: var.endswith(ending_with))
        )

        with pl.Config(fmt_str_lengths=200, tbl_rows=-1):
            print(filtered_feats)

        filtered_feats = None
        feat_defs = None

    @staticmethod
    def find_index(lst: list[Any], item: Any) -> Union[int, None]:
        """
        Finds the index of an item in a list.

        Args:
        - lst (list): List to search.
        - item (Any): Item to find in the list.

        Returns:
        - int | None: Index of the item if found, otherwise None.
        """
        try:
            return lst.index(item)
        except ValueError:
            return None

    @staticmethod
    def dtype_to_str(dtype: pl.DataType) -> str:
        """
        Converts Polars data type to string representation.

        Args:
        - dtype (pl.DataType): Polars data type.

        Returns:
        - str: String representation of the data type.
        """
        dtype_map = {
            pl.Decimal: "Decimal",
            pl.Float32: "Float32",
            pl.Float64: "Float64",
            pl.UInt8: "UInt8",
            pl.UInt16: "UInt16",
            pl.UInt32: "UInt32",
            pl.UInt64: "UInt64",
            pl.Int8: "Int8",
            pl.Int16: "Int16",
            pl.Int32: "Int32",
            pl.Int64: "Int64",
            pl.Date: "Date",
            pl.Datetime: "Datetime",
            pl.Duration: "Duration",
            pl.Time: "Time",
            pl.Array: "Array",
            pl.List: "List",
            pl.Struct: "Struct",
            pl.String: "String",
            pl.Categorical: "Categorical",
            pl.Enum: "Enum",
            pl.Utf8: "Utf8",
            pl.Binary: "Binary",
            pl.Boolean: "Boolean",
            pl.Null: "Null",
            pl.Object: "Object",
            pl.Unknown: "Unknown",
        }

        return dtype_map.get(dtype)

    @staticmethod
    def find_feat_occur(regex_path: str, ending_with: str) -> pl.DataFrame:
        """
        Finds occurrences of features ending with a specific string in Parquet files.

        Args:
        - regex_path (str): Regular expression to match Parquet file paths.
        - ending_with (str): Ending to filter feature names.

        Returns:
        - pl.DataFrame: DataFrame containing feature definitions, data types, and file locations.
        """
        feat_defs: pl.DataFrame = pl.read_csv(dataPath / "feature_definitions.csv").filter(
            pl.col("Variable").apply(lambda var: var.endswith(ending_with))
        )
        feat_defs.sort(by=["Variable"])

        feats: list[pl.String] = feat_defs["Variable"].to_list()
        feats.sort()

        occurrences: list[list] = [[set(), set()] for _ in range(feat_defs.height)]

        for path in glob(str(regex_path)):
            df_schema: dict = pl.read_parquet_schema(path)

            for feat, dtype in df_schema.items():
                index: int = Utility.find_index(feats, feat)
                if index != None:
                    occurrences[index][0].add(Utility.dtype_to_str(dtype))
                    occurrences[index][1].add(Path(path).stem)

        data_types: list[str] = [None] * feat_defs.height
        file_locs: list[str] = [None] * feat_defs.height

        for i, feat in enumerate(feats):
            data_types[i] = list(occurrences[i][0])
            file_locs[i] = list(occurrences[i][1])

        feat_defs = feat_defs.with_columns(pl.Series(data_types).alias("Data_Type(s)"))
        feat_defs = feat_defs.with_columns(pl.Series(file_locs).alias("File_Loc(s)"))

        return feat_defs

    def reduce_memory_usage(df: pl.DataFrame, name) -> pl.DataFrame:
        """
        Reduces memory usage of a DataFrame by converting column types.

        Args:
        - df (pl.DataFrame): DataFrame to optimize.
        - name (str): Name of the DataFrame.

        Returns:
        - pl.DataFrame: Optimized DataFrame.
        """
        print(
            f"Memory usage of dataframe \"{name}\" is {round(df.estimated_size('mb'), 4)} MB."
        )

        int_types = [
            pl.Int8,
            pl.Int16,
            pl.Int32,
            pl.Int64,
            pl.UInt8,
            pl.UInt16,
            pl.UInt32,
            pl.UInt64,
        ]
        float_types = [pl.Float32, pl.Float64]

        for col in df.columns:
            col_type = df[col].dtype
            if col_type in int_types + float_types:
                c_min = df[col].min()
                c_max = df[col].max()

                if c_min is not None and c_max is not None:
                    if col_type in int_types:
                        if c_min >= 0:
                            if (
                                c_min >= np.iinfo(np.uint8).min
                                and c_max <= np.iinfo(np.uint8).max
                            ):
                                df = df.with_columns(df[col].cast(pl.UInt8))
                            elif (
                                c_min >= np.iinfo(np.uint16).min
                                and c_max <= np.iinfo(np.uint16).max
                            ):
                                df = df.with_columns(df[col].cast(pl.UInt16))
                            elif (
                                c_min >= np.iinfo(np.uint32).min
                                and c_max <= np.iinfo(np.uint32).max
                            ):
                                df = df.with_columns(df[col].cast(pl.UInt32))
                            elif (
                                c_min >= np.iinfo(np.uint64).min
                                and c_max <= np.iinfo(np.uint64).max
                            ):
                                df = df.with_columns(df[col].cast(pl.UInt64))
                        else:
                            if (
                                c_min >= np.iinfo(np.int8).min
                                and c_max <= np.iinfo(np.int8).max
                            ):
                                df = df.with_columns(df[col].cast(pl.Int8))
                            elif (
                                c_min >= np.iinfo(np.int16).min
                                and c_max <= np.iinfo(np.int16).max
                            ):
                                df = df.with_columns(df[col].cast(pl.Int16))
                            elif (
                                c_min >= np.iinfo(np.int32).min
                                and c_max <= np.iinfo(np.int32).max
                            ):
                                df = df.with_columns(df[col].cast(pl.Int32))
                            elif (
                                c_min >= np.iinfo(np.int64).min
                                and c_max <= np.iinfo(np.int64).max
                            ):
                                df = df.with_columns(df[col].cast(pl.Int64))
                    elif col_type in float_types:
                        if (
                            c_min > np.finfo(np.float32).min
                            and c_max < np.finfo(np.float32).max
                        ):
                            df = df.with_columns(df[col].cast(pl.Float32))

        print(
            f"Memory usage of dataframe \"{name}\" became {round(df.estimated_size('mb'), 4)} MB."
        )

        return df

    def to_pandas(df: pl.DataFrame, cat_cols: list[str] = None) -> (pd.DataFrame, list[str]):  # type: ignore
        """
        Converts a Polars DataFrame to a Pandas DataFrame.

        Args:
        - df (pl.DataFrame): Polars DataFrame to convert.
        - cat_cols (list[str]): List of categorical columns. Default is None.

        Returns:
        - (pd.DataFrame, list[str]): Tuple containing the converted Pandas DataFrame and categorical columns.
        """
        df: pd.DataFrame = df.to_pandas()

        if cat_cols is None:
            cat_cols = list(df.select_dtypes("object").columns)

        df[cat_cols] = df[cat_cols].astype("str")

        return df, cat_cols


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

                if isnull > 0.95:
                    df = df.drop(col)

        for col in df.columns:
            if (col not in ["target", "case_id", "WEEK_NUM"]) & (df[col].dtype == pl.String):
                freq = df[col].n_unique()

                if (freq == 1) | (freq > 200):
                    df = df.drop(col)

        return df

class Aggregator:
    @staticmethod
    def num_expr(df):
        cols = [col for col in df.columns if col[-1] in ("P", "A")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        return expr_max

    @staticmethod
    def date_expr(df):
        cols = [col for col in df.columns if col[-1] in ("D",)]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        return expr_max

    @staticmethod
    def str_expr(df):
        cols = [col for col in df.columns if col[-1] in ("M",)]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        return expr_max

    @staticmethod
    def other_expr(df):
        cols = [col for col in df.columns if col[-1] in ("T", "L")]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        return expr_max

    @staticmethod
    def count_expr(df):
        cols = [col for col in df.columns if "num_group" in col]
        expr_max = [pl.max(col).alias(f"max_{col}") for col in cols]
        return expr_max

    @staticmethod
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

def predict(dataPath, pkl_path, ohe=None):
    
    with open(pkl_path, 'r') as f:
        loaded_data = json.load(f)


    cat_cols = loaded_data['cat_col']
    train_col = loaded_data['df_train_col']

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
        ((pl.col('max_contractmaturitydate_151D') - pl.col('max_contractdate_551D')).dt.total_days()).alias('contract_duration_days')
    ).with_columns(
        ((pl.col('max_lastupdate_260D') - pl.col('max_contractdate_551D')).dt.total_days()).alias('last_update_duration_days')
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
        .alias('max_dbddpd_booleanP')
    )

    test_static = test_static.with_columns(
        pl.when(
            (pl.col('maxdbddpdlast1m_3658939P') <= 0) &
            (pl.col('maxdbddpdtollast12m_3658940P') <= 0) &
            (pl.col('maxdbddpdtollast6m_4187119P') <= 0)
        )
        .then(1)
        .otherwise(0)
        .alias('max_pays_debt_on_timeP')
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
    df_test = df_test.pipe(Pipeline.handle_dates).pipe(Utility.reduce_memory_usage, "df_train")

    df_test = df_test.select([col for col in train_col if col != "target"])

    df_test, cat_cols = to_pandas(df_test, cat_cols)
        
    X_test = df_test.drop(columns=["WEEK_NUM"])
    case_ids = X_test["case_id"]
    X_test = X_test.set_index("case_id")
    
    model = pickle.loads(base64.b64decode(loaded_data['model'].encode('utf-8')))
    score = model.predict_proba(X_test)[:, 1]
    
    df_submission = pd.DataFrame({
        "case_id": case_ids,
        "score": score
    }).set_index('case_id')

    return df_submission

data_path = "../../../data/home-credit-credit-risk-model-stability/parquet_files/test"
pkl_path = "data.json"
df_submission = predict(data_path, pkl_path)