"""Conversational data-analyst: answers questions about a CSV from Google Drive.

The agent gets one fixed, closed set of pandas-backed tools (load, describe,
filter, group-by aggregate). There is no shell, no sandbox and no arbitrary
code execution: every number in an answer comes from one of these tools.
"""

from __future__ import annotations

import asyncio
import base64
import json
from io import StringIO
from typing import Annotated, Literal

import pandas as pd
from pydantic import BaseModel, Field

from mistralai import agents

_FRAMES: dict[str, pd.DataFrame] = {}
_FILE_NAMES: dict[str, str] = {}


class FilterCondition(BaseModel):
    """One AND-ed row filter."""

    column: Annotated[str, Field(description="Column name to filter on.")]
    operator: Annotated[
        Literal[
            "==",
            "!=",
            ">",
            ">=",
            "<",
            "<=",
            "in",
            "not_in",
            "contains",
            "starts_with",
            "is_null",
            "not_null",
        ],
        Field(description="Comparison operator."),
    ]
    value: Annotated[
        str | int | float | list[str | int | float] | None,
        Field(
            description=(
                "Comparison value. A list is required for in/not_in. Ignored for "
                "is_null/not_null."
            )
        ),
    ] = None


class AggregationSpec(BaseModel):
    """One aggregation over one column."""

    column: Annotated[
        str,
        Field(description='Column to aggregate. Use "_rows" (combined with count) for the row count.'),
    ]
    func: Annotated[
        Literal["count", "sum", "mean", "median", "min", "max", "std", "nunique", "first", "last"],
        Field(description="Aggregation function."),
    ]


def _records(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    view = df.head(limit) if limit is not None else df
    return json.loads(view.to_json(orient="records", date_format="iso", double_precision=6))


def _load_frame(ctx: agents.ToolCallbackContext) -> tuple[pd.DataFrame, str]:
    key = ctx.session.id
    frame = _FRAMES.get(key)
    if frame is None:
        raise agents.ToolError(
            "No dataset is loaded for this session (the dataset memory is gone, most likely "
            "because the worker restarted). Download the CSV again with the Google Drive "
            "connector and call load_csv first."
        )
    return frame, _FILE_NAMES.get(key, "dataset")


def _numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _datetime(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)


def _pair(series: pd.Series, value: str | int | float) -> tuple[pd.Series, object]:
    if _numeric(series):
        return series, float(value)
    if _datetime(series):
        return series, pd.to_datetime(value)
    return (
        series.astype("string").str.strip().str.lower(),
        str(value).strip().lower(),
    )


def _value_list(series: pd.Series, values: list[str | int | float]) -> list:
    if _numeric(series):
        return [float(v) for v in values]
    if _datetime(series):
        return [pd.to_datetime(v) for v in values]
    return [str(v).strip().lower() for v in values]


def _conditions(items: list[FilterCondition]) -> list[FilterCondition]:
    return [
        item if isinstance(item, FilterCondition) else FilterCondition.model_validate(item)
        for item in items
    ]


def _apply_filters(df: pd.DataFrame, conditions: list[FilterCondition]) -> pd.DataFrame:
    for condition in conditions:
        column = condition.column
        if column not in df.columns:
            known = ", ".join(map(str, df.columns))
            raise agents.ToolError(f"Unknown column {column!r}. Available columns: {known}.")
        series = df[column]
        operator = condition.operator
        if operator == "is_null":
            mask = series.isna()
        elif operator == "not_null":
            mask = series.notna()
        else:
            value = condition.value
            if value is None:
                raise agents.ToolError(
                    f"Operator {operator!r} on column {column!r} needs a value "
                    "(use is_null/not_null to test for missing values)."
                )
            if operator in ("in", "not_in"):
                if not isinstance(value, list):
                    value = [value]
                if _numeric(series) or _datetime(series):
                    compared = series
                else:
                    compared = series.astype("string").str.strip().str.lower()
                expected = _value_list(series, value)
            else:
                compared, expected = _pair(series, value)
            match operator:
                case "==":
                    mask = compared == expected
                case "!=":
                    mask = compared != expected
                case ">":
                    mask = compared > expected
                case ">=":
                    mask = compared >= expected
                case "<":
                    mask = compared < expected
                case "<=":
                    mask = compared <= expected
                case "in":
                    mask = compared.isin(expected)
                case "not_in":
                    mask = ~compared.isin(expected)
                case "contains":
                    mask = compared.astype("string").str.contains(
                        str(expected), case=False, regex=False
                    )
                case "starts_with":
                    mask = compared.astype("string").str.startswith(str(expected), na=False)
                case _:
                    raise agents.ToolError(f"Unsupported operator {operator!r}.")
        df = df[mask.fillna(False)]
    return df


def _check_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        known = ", ".join(map(str, df.columns))
        raise agents.ToolError(
            f"Unknown column(s): {', '.join(missing)}. Available columns: {known}."
        )


@agents.tool
async def load_csv(
    ctx: agents.ToolCallbackContext,
    /,
    csv_base64: Annotated[
        str | None,
        Field(
            description="Base64-encoded CSV text, as returned by the Drive connector's "
            "download_file_content. Pass either this or csv_text."
        ),
    ] = None,
    csv_text: Annotated[
        str | None,
        Field(description="Raw CSV text. Pass either this or csv_base64."),
    ] = None,
    file_name: Annotated[str, Field(description="File title, used to label results.")] = "dataset",
    separator: Annotated[
        str | None,
        Field(description='Column separator, e.g. "," or ";". Leave out to auto-detect.'),
    ] = None,
    decimal: Annotated[
        str, Field(description="Decimal mark in numbers, e.g. '.' or, for European CSVs, ','.")
    ] = ".",
    thousands: Annotated[
        str | None,
        Field(description='Thousands separator, e.g. "." for European CSVs with decimal="\',\'".'),
    ] = None,
    encoding: Annotated[str, Field(description="Text encoding of the CSV.")] = "utf-8",
    infer_dates: Annotated[
        bool, Field(description="Parse date-like text columns to datetimes.")
    ] = True,
) -> dict:
    """Load a CSV from Google Drive as this session's dataset.

    Call this right after download_file_content from the google_drive_mcp
    connector: pass its base64 content as csv_base64, or the raw text as
    csv_text. Loads exactly one dataset for this conversation; loading again
    replaces the previous one.
    """
    key = ctx.session.id
    if (csv_base64 is None) == (csv_text is None):
        raise agents.ToolError("Pass exactly one of csv_base64 or csv_text.")
    if csv_base64 is not None:
        try:
            text = base64.b64decode(csv_base64, validate=False).decode(encoding)
        except Exception as error:
            raise agents.ToolError(
                f"Could not decode csv_base64 ({error}); download the file again."
            ) from error
    else:
        text = csv_text or ""
    if not text.strip():
        raise agents.ToolError("The CSV is empty. Check the file in Google Drive.")
    try:
        if separator is None:
            frame = pd.read_csv(
                StringIO(text),
                sep=None,
                engine="python",
                decimal=decimal,
                thousands=thousands,
                encoding=encoding,
            )
        else:
            frame = pd.read_csv(
                StringIO(text),
                sep=separator,
                decimal=decimal,
                thousands=thousands,
                encoding=encoding,
            )
    except Exception:
        if encoding.lower().replace("-", "") != "utf8":
            raise agents.ToolError(
                f"Could not parse the CSV with encoding {encoding!r}."
            ) from None
        frame = pd.read_csv(StringIO(text), sep=None, engine="python", encoding="latin-1")
    if frame.columns.size == 0:
        raise agents.ToolError(
            "The CSV parsed to zero columns. If the file is not a plain CSV, export it "
            "(Google Sheets: download with export_mime_type='text/csv') and try again."
        )
    frame.columns = [str(column).strip() for column in frame.columns]
    if infer_dates:
        for column in frame.select_dtypes(include="object").columns:
            parsed = pd.to_datetime(frame[column], errors="coerce", utc=False)
            if parsed.notna().mean() > 0.8:
                frame[column] = parsed
    if len(frame) == 0:
        raise agents.ToolError(
            "The CSV parsed but has zero rows. Check the separator and the file content."
        )
    _FRAMES[key] = frame
    _FILE_NAMES[key] = file_name
    return {
        "file_name": file_name,
        "rows": int(len(frame)),
        "columns": list(map(str, frame.columns)),
        "dtypes": {str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        "null_cells": int(frame.isna().sum().sum()),
        "preview_rows": _records(frame, limit=3),
    }


@agents.tool
async def describe_columns(ctx: agents.ToolCallbackContext, /) -> dict:
    """Describe the loaded dataset: per-column type, missing values, ranges and top values.

    Use this as the first look at the data for a question, and whenever the
    user asks what the file contains.
    """
    frame, file_name = _load_frame(ctx)
    columns: list[dict] = []
    for column in frame.columns:
        series = frame[column]
        info: dict = {
            "column": str(column),
            "dtype": str(series.dtype),
            "non_null": int(series.notna().sum()),
            "nulls": int(series.isna().sum()),
            "unique": int(series.nunique()),
        }
        if _numeric(series) and series.notna().any():
            info.update(
                {
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                }
            )
        elif _datetime(series) and series.notna().any():
            info.update({"min": str(series.min()), "max": str(series.max())})
        else:
            counts = series.value_counts(dropna=True).head(5)
            info["top_values"] = [
                {"value": str(value), "count": int(count)} for value, count in counts.items()
            ]
        columns.append(info)
    return {
        "file_name": file_name,
        "rows": int(len(frame)),
        "column_count": int(frame.columns.size),
        "columns": columns,
    }


@agents.tool
async def filter_rows(
    ctx: agents.ToolCallbackContext,
    /,
    conditions: Annotated[
        list[FilterCondition],
        Field(description="Row conditions, all AND-ed together."),
    ],
    columns: Annotated[
        list[str] | None,
        Field(description="Return only these columns; all if left out."),
    ] = None,
    sample_limit: Annotated[
        int, Field(description="Maximum number of matching rows to return (up to 100).")
    ] = 20,
) -> dict:
    """Filter the loaded dataset's rows and return the matches and the matched count.

    Use it to list rows that meet conditions. For counts or sums per group,
    prefer groupby_aggregate in one call instead.
    """
    frame, file_name = _load_frame(ctx)
    sample_limit = max(1, min(sample_limit, 100))
    if columns:
        _check_columns(frame, columns)
    matched = _apply_filters(frame, _conditions(conditions))
    if columns:
        matched = matched[columns]
    return {
        "file_name": file_name,
        "matched_rows": int(len(matched)),
        "total_rows": int(len(frame)),
        "sample_limit": sample_limit,
        "truncated": len(matched) > sample_limit,
        "rows": _records(matched, limit=sample_limit),
    }


@agents.tool
async def groupby_aggregate(
    ctx: agents.ToolCallbackContext,
    /,
    group_by: Annotated[list[str], Field(description="Column(s) to group by.")],
    aggregations: Annotated[
        list[AggregationSpec],
        Field(
            description="Aggregations to compute per group, e.g. "
            "{'column': 'revenue', 'func': 'sum'}."
        ),
    ],
    filters: Annotated[
        list[FilterCondition] | None,
        Field(description="Row conditions applied before grouping, all AND-ed."),
    ] = None,
    order_by: Annotated[
        str | None,
        Field(
            description="Sort the result by this output column ('<func>_<column>' "
            "or a group column)."
        ),
    ] = None,
    descending: Annotated[bool, Field(description="Sort order for order_by.")] = True,
    limit: Annotated[
        int, Field(description="Maximum number of groups to return (up to 200).")
    ] = 50,
) -> dict:
    """Group the loaded dataset and compute aggregations, returning real numbers.

    Use this for per-group totals, averages, counts and the like, optionally
    with row filters applied first. The result columns are named '<func>_<column>'.
    """
    frame, file_name = _load_frame(ctx)
    limit = max(1, min(limit, 200))
    _check_columns(frame, group_by)
    if not aggregations:
        raise agents.ToolError("Pass at least one aggregation, e.g. count over _rows.")
    named: dict[str, pd.NamedAgg] = {}
    for raw_spec in aggregations:
        spec = (
            raw_spec
            if isinstance(raw_spec, AggregationSpec)
            else AggregationSpec.model_validate(raw_spec)
        )
        name = "row_count" if spec.column == "_rows" else f"{spec.func}_{spec.column}"
        if spec.column == "_rows":
            if spec.func not in ("count", "size"):
                raise agents.ToolError('Column "_rows" only supports count.')
            named[name] = pd.NamedAgg(column=group_by[0], aggfunc="size")
        else:
            _check_columns(frame, [spec.column])
            named[name] = pd.NamedAgg(column=spec.column, aggfunc=spec.func)
    source = _apply_filters(frame, _conditions(filters)) if filters else frame
    if len(source) == 0:
        return {
            "file_name": file_name,
            "rows_after_filters": 0,
            "groups": [],
            "note": "No rows matched the filters.",
        }
    grouped = (
        source.groupby(list(group_by), dropna=False, observed=True).agg(**named).reset_index()
    )
    if order_by:
        if order_by not in grouped.columns:
            raise agents.ToolError(
                f"Unknown order_by {order_by!r}. Use a group column or an output column: "
                f"{', '.join(map(str, named))}."
            )
        grouped = grouped.sort_values(order_by, ascending=not descending)
    return {
        "file_name": file_name,
        "rows_after_filters": int(len(source)),
        "group_count": int(len(grouped)),
        "groups_returned": min(limit, len(grouped)),
        "truncated": len(grouped) > limit,
        "groups": _records(grouped, limit=limit),
    }


_INSTRUCTIONS = """\
Je bent een conversationele data-analist. De gebruiker geeft een link naar een \
CSV-bestand in Google Drive en stelt daarna vragen in gewone taal over die data. \
Je antwoordt telkens met echte cijfers uit de data, in de taal van de gebruiker \
(standaard Nederlands).

Bestand ophalen en laden (bij de eerste vraag of bij een nieuw bestand):
1. Haal de file_id uit de link die de gebruiker gaf (verander de link niet): \
https://docs.google.com/.../d/<file_id>/... .
2. Gebruik de google_drive_mcp-connector. Ken je de file_id niet of lukt het niet, \
zoek dan met search_files op bestandsnaam (vraag de gebruiker pas om een naam als de \
zoekactie niets vindt) en gebruik daarna download_file_content met die file_id. \
Kies expliciet export_mime_type="text/csv" als het bestand een Google Sheet is.
3. Roep daarna direct load_csv aan en geef de base64-inhoud door als csv_base64 \
(als je alleen platte tekst hebt: csv_text) plus de bestandsnaam als file_name. \
JSON of XLSX worden niet ondersteund: vraag de gebruiker om een CSV.
4. Herlaad het bestand niet als het al geladen is; herslaad alleen als een tool zegt \
dat de dataset niet meer beschikbaar is.

Vragen beantwoorden:
- Gebruik uitsluitend de vaste tools op de geladen dataset, in deze volgorde van \
voorkeur: describe_columns om te zien wat er in de data zit; groupby_aggregate voor \
elke per-groep count, som, gemiddelde enzovoort (row filters meegeven met de \
filters-parameter); filter_rows om rijen op te sommeren die aan voorwaarden voldoen.
- Gebruik kolom- en waardenamen precies zoals ze in de data staan; controleer ze eerst \
met describe_columns. Bedenk, interpoleer of verzin nooit een getal: elk cijfer in je \
antwoord komt uit een toolresultaat. Als een vraag niet mogelijk is met de data of de \
tools, zeg dat en leg uit waarom.
- Bij grote resultaten: toon de relevante groepen of rijen en geef wel de exacte \
totalen en het aandeel ten opzichte van het geheel (bij voorkeur in %).
- Vermeld kort de gebruikte filters en aggregaties, zodat de gebruiker kan nagaan wat \
er berekend is. Vervolgvragen in hetzelfde gesprek gaan automatisch over de geladen \
dataset en de eerdere bevindingen; de gebruiker hoeft niets opnieuw uit te leggen.
- Je hebt bewust geen shell, sandbox of code-uitvoering. Gebruik alleen deze vaste \
tools en de Google Drive-connector.
"""

agent = agents.Agent(
    model="mistral-large-4",
    name="data-analyst",
    instructions=_INSTRUCTIONS,
    tools=[load_csv, describe_columns, filter_rows, groupby_aggregate],
    connectors=[agents.connector("google_drive_mcp")],
)


async def main() -> None:
    print(
        "This is a conversational data-analyst chatbot. Open a chat with: "
        "uv run --env-file .env agents chat"
    )


if __name__ == "__main__":
    asyncio.run(main())
