"""Extraction utilities for local datasets and SQL databases."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_FILE_EXTENSIONS = frozenset({".csv", ".json", ".parquet", ".xlsx", ".xls"})


def extract(path: str | Path, **read_options: Any) -> pd.DataFrame:
    """Read a supported local dataset into a DataFrame.

    Args:
        path: Path to a CSV, JSON, Parquet, XLSX, or XLS dataset.
        **read_options: Options passed to the corresponding pandas reader, such
            as ``encoding`` for CSV files or ``sheet_name`` for Excel files.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        IsADirectoryError: If ``path`` is a directory.
        ValueError: If the file extension is unsupported.
    """
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found: {source}")
    if not source.is_file():
        raise IsADirectoryError(f"Dataset path must be a file: {source}")

    extension = source.suffix.lower()
    readers = {
        ".csv": pd.read_csv,
        ".json": pd.read_json,
        ".parquet": pd.read_parquet,
        ".xlsx": pd.read_excel,
        ".xls": pd.read_excel,
    }
    reader = readers.get(extension)
    if reader is None:
        supported = ", ".join(sorted(SUPPORTED_FILE_EXTENSIONS))
        raise ValueError(f"Unsupported dataset extension '{extension}'. Supported: {supported}")

    return reader(source, **read_options)


def extract_sql(
    query: str,
    connection: Any,
    *,
    params: Mapping[str, Any] | tuple[Any, ...] | None = None,
    **read_options: Any,
) -> pd.DataFrame:
    """Run a SQL query and return its result as a DataFrame.

    ``connection`` can be a DB-API connection (for example, from ``sqlite3``),
    a SQLAlchemy engine/connection, or a SQLAlchemy connection URL. For
    PostgreSQL, MySQL, and similar databases, install SQLAlchemy plus the
    appropriate database driver.

    Pass dynamic values through ``params`` instead of interpolating them into
    the query string. Placeholder syntax depends on the selected SQL driver.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("A non-empty SQL query is required")
    if connection is None:
        raise ValueError("A database connection, engine, or connection URL is required")

    return pd.read_sql_query(query, con=connection, params=params, **read_options)
