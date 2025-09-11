import datetime
from typing import Any, Callable, List, Optional, Type

import polars as pl
import pytest

from fenic.api.dataframe import DataFrame
from fenic.api.functions import col
from fenic.api.functions.dt import (
    current_date,
    current_timestamp,
    date_add,
    date_format,
    date_sub,
    date_trunc,
    datediff,
    day,
    hour,
    millisecond,
    minute,
    month,
    now,
    second,
    timestamp_add,
    timestamp_diff,
    to_date,
    to_timestamp,
    year,
)
from fenic.core.types import (
    ColumnField,
    DateType,
    IntegerType,
    StringType,
    TimestampType,
)

# Full timestamp results.
TS_FULL_RESULTS = [
    datetime.datetime(2025, 1, 1, 1, 0),
    datetime.datetime(2025, 2, 2, 2, 20, 0, 200000),
    datetime.datetime(2025, 3, 3, 15, 33, 30, 300000)
]

TS_FULL_RESULTS_PLUS1 = [
    datetime.datetime(2025, 1, 2, 1, 0),
    datetime.datetime(2025, 2, 3, 2, 20, 0, 200000),
    datetime.datetime(2025, 3, 4, 15, 33, 30, 300000)
]

TS_FULL_RESULTS_MINUS1 = [
    datetime.datetime(2024, 12, 31, 1, 0),
    datetime.datetime(2025, 2, 1, 2, 20, 0, 200000),
    datetime.datetime(2025, 3, 2, 15, 33, 30, 300000)
]

TS_FULL_RESULTS_PLUS1YEAR = [
    datetime.datetime(2026, 1, 1, 1, 0),
    datetime.datetime(2026, 2, 2, 2, 20, 0, 200000),
    datetime.datetime(2026, 3, 3, 15, 33, 30, 300000)
]

# Full date results.
DT_FULL_RESULTS = [
    datetime.date(2025, 1, 1),
    datetime.date(2025, 1, 2),
    datetime.date(2025, 1, 3)
]

DT_FULL_RESULTS_MINUS1 = [
    datetime.date(2024, 12, 31),
    datetime.date(2025, 1, 1),
    datetime.date(2025, 1, 2)
]

DT_FULL_RESULTS_PLUS1 = [
    datetime.date(2025, 1, 2),
    datetime.date(2025, 1, 3),
    datetime.date(2025, 1, 4)
]

DT_FULL_RESULTS_PLUS1YEAR = [
    datetime.date(2026, 1, 1),
    datetime.date(2026, 1, 2),
    datetime.date(2026, 1, 3)
]


@pytest.fixture
def df_with_date_types(local_session):
    pl_df = pl.DataFrame({
        "col1": [1, 2, 3],
        "date_str": [
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
        ],
        "date": [
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
        ],
        "date_plus": [
            "2025-01-10",
            "2025-01-20",
            "2025-01-30",
        ],
        "ts_str": [
            "2025-01-01T1:00:00.000",
            "2025-02-02T02:20:00.200",
            "2025-03-03T15:33:30.300",
        ]
    },
    schema={
        "col1": pl.Int64,
        "date_str": pl.String,
        "date": pl.Date,
        "date_plus": pl.Date,
        "ts_str": pl.String,
    })

    pl_df = pl_df.with_columns(
        pl.col("ts_str").str.strptime(pl.Datetime, format="%Y-%m-%dT%H:%M:%S.%3f").alias("ts")
    )

    pl_df = pl_df.with_columns(
        pl.col("ts_str").str.strptime(pl.Date, format="%Y-%m-%dT%H:%M:%S.%3f").alias("dt")
    )

    return local_session.create_dataframe(pl_df)

class TestDateFunctions:
    def test_year(self, df_with_date_types):
        """Test the year function."""
        default_year_results = [2025]*3
        self._verify_results(df_with_date_types, "ts", year, IntegerType, "year", default_year_results)
        self._verify_results(df_with_date_types, "date", year, IntegerType, "year", default_year_results)
        self._verify_error(df_with_date_types, "col1", year, TypeError)

    def test_month(self, df_with_date_types):
        """Test the month function."""
        self._verify_results(df_with_date_types, "ts", month, IntegerType, "month", [1, 2, 3])
        self._verify_results(df_with_date_types, "date", month, IntegerType, "month", [1, 1, 1])
        self._verify_error(df_with_date_types, "col1", month, TypeError)

    def test_day(self, df_with_date_types):
        """Test the day function."""
        self._verify_results(df_with_date_types, "ts", day, IntegerType, "day", [1, 2, 3])
        self._verify_results(df_with_date_types, "date", day, IntegerType, "day", [1, 2, 3])
        self._verify_error(df_with_date_types, "col1", day, TypeError)

    def test_hour(self, df_with_date_types):
        """Test the hour function."""
        self._verify_results(df_with_date_types, "ts", hour, IntegerType, "hour", [1, 2, 15])
        self._verify_results(df_with_date_types, "date", hour, IntegerType, "hour", [0, 0, 0])
        self._verify_error(df_with_date_types, "col1", hour, TypeError)

    def test_minute(self, df_with_date_types):
        """Test the minute function."""
        self._verify_results(df_with_date_types, "ts", minute, IntegerType, "minute", [0, 20, 33])
        self._verify_results(df_with_date_types, "date", minute, IntegerType, "minute", [0, 0, 0])
        self._verify_error(df_with_date_types, "col1", minute, TypeError)

    def test_second(self, df_with_date_types):
        """Test the second function."""
        self._verify_results(df_with_date_types, "ts", second, IntegerType, "second", [0, 0, 30])
        self._verify_results(df_with_date_types, "date", second, IntegerType, "second", [0, 0, 0])
        self._verify_error(df_with_date_types, "col1", second, TypeError)

    def test_millisecond(self, df_with_date_types):
        """Test the millisecond function."""
        self._verify_results(df_with_date_types, "ts", millisecond, IntegerType, "millisecond", [0, 200, 300])
        self._verify_results(df_with_date_types, "date", millisecond, IntegerType, "millisecond", [0, 0, 0])
        self._verify_error(df_with_date_types, "col1", millisecond, TypeError)

    def test_to_timestamp(self, df_with_date_types):
        """Test the to_timestamp function."""
        self._verify_results(df_with_date_types, "ts_str", to_timestamp, TimestampType, "ts2", TS_FULL_RESULTS)
        self._verify_error(df_with_date_types, "col1", to_timestamp, TypeError)

    def test_to_date(self, df_with_date_types, local_session):
        """Test the to_date function."""
        self._verify_results(df_with_date_types, "date_str", to_date, DateType, "dt2", DT_FULL_RESULTS)
        self._verify_error(df_with_date_types, "col1", to_date, TypeError)

        df = local_session.create_dataframe(pl.DataFrame({
            "date_str": ["01-27-2025", "11-02-2025"],
        }))
        df = df.select(to_date(col("date_str"), format="MM-dd-yyyy").alias("dt2"))
        self._verify_results_on_df(df, DateType, "dt2", [
            datetime.date(2025, 1, 27),
            datetime.date(2025, 11, 2),
        ])

    def test_current_date_functions(self, df_with_date_types):
        """Test the current date and time functions."""
        now_year = datetime.datetime.now().year
        now_year_results = [now_year]*3

        # now
        df = df_with_date_types.with_column("now", now())
        df = df.select(col("now"))
        assert df.schema.column_fields == [ColumnField("now", TimestampType)]

        df = df.select(year(col("now")).alias("now_year"))
        self._verify_results_on_df(df, IntegerType, "now_year", now_year_results)

        # current_timestamp
        df = df_with_date_types.with_column("current_timestamp", current_timestamp())
        df = df.select(col("current_timestamp"))
        assert df.schema.column_fields == [ColumnField("current_timestamp", TimestampType)]

        df = df.select(year(col("current_timestamp")).alias("current_timestamp_year"))
        self._verify_results_on_df(df, IntegerType, "current_timestamp_year", now_year_results)

        # current_date
        df = df_with_date_types.with_column("current_date", current_date())
        df = df.select(col("current_date"))
        assert df.schema.column_fields == [ColumnField("current_date", DateType)]

        df = df.select(year(col("current_date")).alias("current_date_year"))
        self._verify_results_on_df(df, IntegerType, "current_date_year", now_year_results)

    def test_date_trunc(self, df_with_date_types):
        """Test the date_trunc function."""
        self._verify_error(df_with_date_types, "ts", date_trunc, ValueError, ["invalid"])
        self._verify_error(df_with_date_types, "col1", date_trunc, TypeError, ["year"])

        self._verify_results(df_with_date_types, "ts", date_trunc, TimestampType, "ts_trunc",  [datetime.datetime(2025, 1, 1, 0, 0)] * 3, ["year"])
        self._verify_results(df_with_date_types, "date", date_trunc, DateType, "dt_trunc", [datetime.date(2025, 1, 1)] * 3, ["year"])
        self._verify_results(df_with_date_types, "ts", date_trunc, TimestampType, "ts_trunc",
            [
                datetime.datetime(2025, 1, 1, 0, 0),
                datetime.datetime(2025, 2, 1, 0, 0),
                datetime.datetime(2025, 3, 1, 0, 0)
            ], ["month"])
        self._verify_results(df_with_date_types, "ts", date_trunc, TimestampType, "ts_trunc",
            [
                datetime.datetime(2025, 1, 1, 0, 0),
                datetime.datetime(2025, 2, 2, 0, 0),
                datetime.datetime(2025, 3, 3, 0, 0)
            ], ["day"])
        self._verify_results(df_with_date_types, "ts", date_trunc, TimestampType, "ts_trunc",
            [
                datetime.datetime(2025, 1, 1, 1, 0),
                datetime.datetime(2025, 2, 2, 2, 0),
                datetime.datetime(2025, 3, 3, 15, 0)
            ], ["hour"])
        self._verify_results(df_with_date_types, "date", date_trunc, DateType, "dt_trunc", [
            datetime.date(2025, 1, 1),
            datetime.date(2025, 1, 2),
            datetime.date(2025, 1, 3)
        ], ["hour"])

    def test_date_add(self, df_with_date_types):
        """Test the date_add function."""
        self._verify_error(df_with_date_types, "col1", date_add, TypeError, ["1"])
        self._verify_error(df_with_date_types, "ts", date_add, TypeError, [col("ts")])

        self._verify_results(df_with_date_types, "ts", date_add, TimestampType, "ts_add", TS_FULL_RESULTS_PLUS1, [1])
        self._verify_results(df_with_date_types, "date", date_add, DateType, "dt_add", DT_FULL_RESULTS_MINUS1, [-1])
        self._verify_results(df_with_date_types, "date", date_sub, DateType, "dt_sub", DT_FULL_RESULTS_PLUS1, [-1])

    def test_timestamp_add(self, df_with_date_types):
        """Test the timestamp_add function."""
        self._verify_error(df_with_date_types, "col1", timestamp_add, TypeError, ["1", "year"])

        self._verify_results(df_with_date_types, "ts", timestamp_add, TimestampType, "ts_add", TS_FULL_RESULTS_PLUS1YEAR, [1, "year"])
        self._verify_results(df_with_date_types, "ts", timestamp_add, TimestampType, "ts_add", TS_FULL_RESULTS_MINUS1, [-1, "day"])
        self._verify_results(df_with_date_types, "date", timestamp_add, DateType, "dt_add", DT_FULL_RESULTS_PLUS1YEAR, [1, "year"])
        self._verify_results(df_with_date_types, "date", timestamp_add, DateType, "dt_add", DT_FULL_RESULTS_MINUS1, [-1, "day"])

    def test_date_format(self, df_with_date_types):
        """Test the date_format function."""
        self._verify_error(df_with_date_types, "col1", date_format, TypeError, ["yyyy-MM-dd"])

        self._verify_results(df_with_date_types, "ts", date_format, StringType, "ts_format", [
            "01-01-2025",
            "02-02-2025",
            "03-03-2025",
        ], ["MM-dd-yyyy"])

        self._verify_results(df_with_date_types, "ts", date_format, StringType, "ts_format",
            [
                '01-01-2025 01:00:00 AM',
                '02-02-2025 02:20:00 AM',
                '03-03-2025 03:33:30 PM'
            ],
            ["MM-dd-yyyy hh:mm:ss a"])
        self._verify_results(df_with_date_types, "date", date_format, StringType, "dt_format",
            [
                "01-01-2025",
                "02-01-2025",
                "03-01-2025",
            ],
            ["dd-MM-yyyy"])

    def test_date_diff(self, df_with_date_types):
        """Test the date_diff function."""
        self._verify_error(df_with_date_types, "col1", datediff, TypeError, [col("col1"), col("date")])
        self._verify_error(df_with_date_types, "ts", datediff, TypeError, [col("ts"), col("col1")])
        self._verify_results(df_with_date_types, "date_plus", datediff, IntegerType, "dt_diff", [9, 18, 27], [col("date")])
        self._verify_results(df_with_date_types, "date", datediff, IntegerType, "dt_diff", [-9, -18, -27], [col("date_plus")])

        df = df_with_date_types.select(
            col("ts"),
            date_add(col("ts"), 10).alias("ts_plus"))
        self._verify_results(df, "ts_plus", datediff, IntegerType, "ts_diff", [10, 10, 10], [col("ts")])

    def test_timestamp_diff(self, df_with_date_types):
        """Test the timestamp_diff function."""
        self._verify_error(df_with_date_types, "col1", timestamp_diff, TypeError, [col("col1"), "day"])
        self._verify_error(df_with_date_types, "ts", timestamp_diff, ValueError, [col("ts"), "invalid"])

        df = df_with_date_types.select(
            col("ts"),
            date_add(col("ts"), 10).alias("ts_plus"))
        self._verify_results(df, "ts_plus", timestamp_diff, IntegerType, "ts_diff", [10, 10, 10], [col("ts"), "day"])
        self._verify_results(df, "ts_plus", timestamp_diff, IntegerType, "ts_diff", [864000000] * 3, [col("ts"), "millisecond"])

        df = df_with_date_types.select(
            col("date"),
            timestamp_add(col("date"), 10, "month").alias("date_plus"))
        self._verify_results(df, "date", timestamp_diff, IntegerType, "dt_diff", [-10, -10, -10], [col("date_plus"), "month"])

    @classmethod
    def _verify_results(
        cls,
        df: DataFrame,
        field_name: str,
        function: Callable,
        expected_type: Type,
        alias: str,
        expected_results: List[Any],
        extra_args: Optional[List[Any]] = None,
    ):
        if extra_args is None:
            extra_args = []
        df = df.select(function(col(field_name), *extra_args).alias(alias))
        cls._verify_results_on_df(df, expected_type, alias, expected_results)

    @classmethod
    def _verify_results_on_df(
        cls,
        df: DataFrame,
        expected_type: Type,
        alias: str,
        expected_results: List[Any],
    ):
        assert df.schema.column_fields == [
            ColumnField(alias, expected_type),
        ]
        assert df.to_polars()[alias].to_list() == expected_results

    @classmethod
    def _verify_error(
        cls,
        df: DataFrame,
        field_name: str,
        function: Callable,
        error_type: Type,
        extra_args: Optional[List[Any]] = None,
    ):
        if extra_args is None:
            extra_args = []
        with pytest.raises(error_type):
            df.select(function(col(field_name), *extra_args))
