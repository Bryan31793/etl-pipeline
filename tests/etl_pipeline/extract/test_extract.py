"""Tests for local and SQL data extraction."""

import sqlite3

import pandas as pd
import pytest

from etl_pipeline.extract.extract import SUPPORTED_FILE_EXTENSIONS, extract, extract_sql


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 2], "name": ["Ana", "Luis"]})


@pytest.fixture
def sqlite_connection():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE customers (id INTEGER, name TEXT, active INTEGER)")
    connection.executemany(
        "INSERT INTO customers VALUES (?, ?, ?)",
        [(1, "Ana", 1), (2, "Luis", 0)],
    )
    yield connection
    connection.close()


def test_supported_file_extensions_are_documented():
    assert SUPPORTED_FILE_EXTENSIONS == {".csv", ".json", ".parquet", ".xlsx", ".xls"}


def test_extract_reads_csv_from_string_path_and_is_case_insensitive(tmp_path, sample_frame):
    dataset = tmp_path / "customers.CSV"
    sample_frame.to_csv(dataset, index=False)

    result = extract(str(dataset))

    pd.testing.assert_frame_equal(result, sample_frame)


def test_extract_forwards_reader_options(tmp_path):
    dataset = tmp_path / "customers.csv"
    dataset.write_text("id;name\n1;Ana\n", encoding="utf-8")

    result = extract(dataset, sep=";")

    pd.testing.assert_frame_equal(result, pd.DataFrame({"id": [1], "name": ["Ana"]}))


def test_extract_reads_json(tmp_path, sample_frame):
    dataset = tmp_path / "customers.json"
    sample_frame.to_json(dataset, orient="records")

    result = extract(dataset, orient="records")

    pd.testing.assert_frame_equal(result, sample_frame)


def test_extract_reads_parquet(tmp_path, sample_frame):
    dataset = tmp_path / "customers.parquet"
    sample_frame.to_parquet(dataset, index=False)

    result = extract(dataset)

    pd.testing.assert_frame_equal(result, sample_frame)


@pytest.mark.parametrize("extension", [".xlsx", ".xls"])
def test_extract_uses_excel_reader_for_excel_extensions(tmp_path, monkeypatch, extension):
    dataset = tmp_path / f"customers{extension}"
    dataset.touch()
    expected = pd.DataFrame({"id": [1]})
    read_excel_calls = []

    def fake_read_excel(path, **options):
        read_excel_calls.append((path, options))
        return expected

    monkeypatch.setattr(pd, "read_excel", fake_read_excel)

    result = extract(dataset, sheet_name="customers")

    pd.testing.assert_frame_equal(result, expected)
    assert read_excel_calls == [(dataset, {"sheet_name": "customers"})]


def test_extract_rejects_missing_dataset(tmp_path):
    dataset = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        extract(dataset)


def test_extract_rejects_directory(tmp_path):
    with pytest.raises(IsADirectoryError, match="must be a file"):
        extract(tmp_path)


@pytest.mark.parametrize("filename", ["customers.xml", "customers"])
def test_extract_rejects_unsupported_extensions(tmp_path, filename):
    dataset = tmp_path / filename
    dataset.write_text("placeholder", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported dataset extension"):
        extract(dataset)


def test_extract_sql_returns_query_results(sqlite_connection):
    result = extract_sql("SELECT id, name FROM customers ORDER BY id", sqlite_connection)

    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame({"id": [1, 2], "name": ["Ana", "Luis"]}),
    )


def test_extract_sql_supports_positional_bound_parameters(sqlite_connection):
    result = extract_sql(
        "SELECT id, name FROM customers WHERE id >= ?",
        sqlite_connection,
        params=(2,),
    )

    pd.testing.assert_frame_equal(result, pd.DataFrame({"id": [2], "name": ["Luis"]}))


def test_extract_sql_supports_named_bound_parameters(sqlite_connection):
    result = extract_sql(
        "SELECT name FROM customers WHERE active = :active",
        sqlite_connection,
        params={"active": 1},
    )

    pd.testing.assert_frame_equal(result, pd.DataFrame({"name": ["Ana"]}))


@pytest.mark.parametrize("query", [None, "", "   ", 42])
def test_extract_sql_requires_a_non_empty_string_query(sqlite_connection, query):
    with pytest.raises(ValueError, match="non-empty SQL query"):
        extract_sql(query, sqlite_connection)


def test_extract_sql_requires_a_connection():
    with pytest.raises(ValueError, match="connection"):
        extract_sql("SELECT 1", None)
