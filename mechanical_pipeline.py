"""Mechanical criteria pipeline utilities.

This module re-implements the key pieces of the notebook
``_Mechanical_Criteria_Pipeline_GPT_2410__Corrected_only.ipynb`` in a
plain-Python, testable form.  It focuses on the mechanical steps that
operate purely on data (loading, normalising, token alignment and
sentence summarisation) and therefore intentionally leaves the LLM
interaction points pluggable.

Usage example
-------------
```
python mechanical_pipeline.py sample_mechanical_input.csv
```
This will read the CSV, apply a lightweight mock correction (so the
pipeline can run offline), build the word alignment table and the
sentence summary table, and finally print small previews of each result.
The functions are written so they can be imported and reused in unit
tests or alternative front-ends.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None  # type: ignore

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

RAW_TEXT_ALIASES: frozenset[str] = frozenset({
    "raw text",
    "raw_text",
    "rawtext",
    "raw-text",
    "response",
})


def _candidate_readers(path: Path) -> Iterator[Tuple[str, Dict[str, object]]]:
    """Yield (label, kwargs) pairs for different ``pandas.read_csv`` strategies."""

    base_kwargs = {
        "encoding": "utf-8-sig",
        "on_bad_lines": "skip",
        "low_memory": False,
    }

    # Auto sniff using python engine
    yield "auto", {**base_kwargs, "sep": None, "engine": "python"}

    # Common explicit delimiters
    for sep in (",", "\t", ";", "|", "^"):
        yield f"sep={sep!r}", {**base_kwargs, "sep": sep}


def load_mechanical_data(
    path: os.PathLike[str] | str,
    *,
    raw_text_column: Optional[str] = None,
    id_column: str = "ID",
) -> pd.DataFrame:
    """Load a CSV and ensure an ``ID`` and ``Raw text`` column exist.

    ``raw_text_column`` may be used to force the column selection; otherwise the
    loader will try a case-insensitive search across ``RAW_TEXT_ALIASES``.
    """

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    last_error: Optional[Exception] = None
    for label, kwargs in _candidate_readers(path):
        try:
            df = pd.read_csv(path, **kwargs)
        except Exception as exc:  # pragma: no cover - depends on file type
            last_error = exc
            continue
        if df.empty:
            continue
        break
    else:  # pragma: no cover - triggers only when *every* reader fails
        raise RuntimeError(f"Unable to read CSV {path!s}: {last_error}")

    # Normalise column names for matching
    normalised = {c.lower().strip(): c for c in df.columns}

    if id_column not in df.columns:
        # Try relaxed search for case-insensitive match
        lowered = {c.lower(): c for c in df.columns}
        if id_column.lower() in lowered:
            df = df.rename(columns={lowered[id_column.lower()]: id_column})
        else:
            raise KeyError(f"ID column '{id_column}' not present in {path!s}")

    if raw_text_column is None:
        for candidate in RAW_TEXT_ALIASES:
            if candidate in normalised:
                raw_text_column = normalised[candidate]
                break
    if raw_text_column is None:
        if "Raw text" in df.columns:
            raw_text_column = "Raw text"
        else:
            raise KeyError(
                "Could not determine the raw text column. Pass ``raw_text_column``"
                " explicitly or add a column matching one of: "
                + ", ".join(sorted(RAW_TEXT_ALIASES))
            )

    df = df.rename(columns={raw_text_column: "Raw text"})
    df = df.copy()
    df["ID"] = df["ID"].astype(str).str.strip()
    df["Raw text"] = df["Raw text"].astype(str).fillna("")
    return df


# ---------------------------------------------------------------------------
# Token counting / mock correction helpers
# ---------------------------------------------------------------------------


def _default_tokenizer_name() -> Optional[str]:
    if tiktoken is None:  # pragma: no cover - depends on optional dep
        return None
    for name in ("o200k_base", "cl100k_base"):
        try:
            tiktoken.get_encoding(name)
            return name
        except Exception:
            continue
    return None


def _token_count(text: str, *, encoding_name: Optional[str]) -> int:
    if not isinstance(text, str) or not text:
        return 0
    if encoding_name and tiktoken is not None:
        try:
            enc = tiktoken.get_encoding(encoding_name)
            return len(enc.encode(text))
        except Exception:  # pragma: no cover - falls back gracefully
            pass
    return len(text.split())  # rough fallback


def add_word_and_token_counts(
    df: pd.DataFrame,
    *,
    text_column: str = "Raw text",
) -> pd.DataFrame:
    """Return a copy of ``df`` with ``WordCount`` and ``TokenCount`` columns."""

    if text_column not in df.columns:
        raise KeyError(f"Column '{text_column}' missing from dataframe")

    df = df.copy()
    encoding_name = _default_tokenizer_name()

    def count_words(text: str) -> int:
        return len([w for w in str(text).split() if w])

    df["WordCount"] = df[text_column].map(count_words)
    df["TokenCount"] = df[text_column].map(
        lambda s: _token_count(str(s), encoding_name=encoding_name)
    )
    return df


TERMINALS = {".", "!", "?", "…", "...", "?!", "!?"}
OPENING_PUNCT = {'"', "“", "‘", "«", "(", "[", "{"}
CLOSING_PUNCT = {'"', "”", "’", "»", ")", "]", "}"}


def mock_correct_text(text: str) -> str:
    """Apply a deterministic, low-touch correction.

    This is intentionally simple: it trims whitespace, upper-cases the first
    alphabetical character and ensures the sentence ends with a period.
    """

    stripped = str(text or "").strip()
    if not stripped:
        return ""
    chars = list(stripped)
    for idx, ch in enumerate(chars):
        if ch.isalpha():
            chars[idx] = ch.upper()
            break
    corrected = "".join(chars)
    if not any(corrected.endswith(term) for term in TERMINALS):
        corrected += "."
    return corrected


def ensure_corrected_column(
    df: pd.DataFrame,
    *,
    raw_column: str = "Raw text",
    corrected_column: str = "Corrected text",
) -> pd.DataFrame:
    """Return ``df`` with a ``corrected_column`` filled.

    If the column already exists it is left untouched.  Otherwise a mock
    correction is produced so the downstream pipeline can be executed offline.
    """

    if raw_column not in df.columns:
        raise KeyError(f"Missing raw text column '{raw_column}'")

    df = df.copy()
    if corrected_column not in df.columns:
        df[corrected_column] = df[raw_column].map(mock_correct_text)
    else:
        df[corrected_column] = df[corrected_column].fillna("").astype(str)
    return df


# ---------------------------------------------------------------------------
# Word-level alignment
# ---------------------------------------------------------------------------

_WORD_RX = re.compile(r"\w", flags=re.UNICODE)


def _simple_tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text or "", flags=re.UNICODE)


def _rebuild_offsets(text: str, tokens: Sequence[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        start = text.find(token, cursor)
        if start < 0:
            start = cursor
        end = start + len(token)
        spans.append((start, end))
        cursor = end
    return spans


@dataclass
class TokenDiff:
    raw_index: Optional[int]
    corr_index: Optional[int]
    raw_token: Optional[str]
    corr_token: Optional[str]
    raw_start: Optional[int]
    raw_end: Optional[int]
    corr_start: Optional[int]
    corr_end: Optional[int]
    op: str
    equal_ci: bool
    error_type: str


def build_word_map(raw_text: str, corr_text: str) -> List[TokenDiff]:
    from difflib import SequenceMatcher

    raw_tokens = _simple_tokenize(raw_text)
    corr_tokens = _simple_tokenize(corr_text)
    raw_spans = _rebuild_offsets(raw_text, raw_tokens)
    corr_spans = _rebuild_offsets(corr_text, corr_tokens)

    sm = SequenceMatcher(
        a=[tok.lower() for tok in raw_tokens],
        b=[tok.lower() for tok in corr_tokens],
        autojunk=False,
    )

    diffs: List[TokenDiff] = []

    def is_word(tok: Optional[str]) -> bool:
        return bool(tok) and bool(_WORD_RX.search(tok))

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                r_idx = i1 + offset
                c_idx = j1 + offset
                diffs.append(
                    TokenDiff(
                        raw_index=r_idx,
                        corr_index=c_idx,
                        raw_token=raw_tokens[r_idx],
                        corr_token=corr_tokens[c_idx],
                        raw_start=raw_spans[r_idx][0],
                        raw_end=raw_spans[r_idx][1],
                        corr_start=corr_spans[c_idx][0],
                        corr_end=corr_spans[c_idx][1],
                        op="equal",
                        equal_ci=(raw_tokens[r_idx] == corr_tokens[c_idx]),
                        error_type="Equal",
                    )
                )
        elif tag == "replace":
            width = min(i2 - i1, j2 - j1)
            for offset in range(width):
                r_idx = i1 + offset
                c_idx = j1 + offset
                raw_tok = raw_tokens[r_idx]
                corr_tok = corr_tokens[c_idx]
                error_type = (
                    "Spelling"
                    if (
                        raw_tok.lower() != corr_tok.lower()
                        and raw_tok.isalpha()
                        and corr_tok.isalpha()
                    )
                    else "Replacement"
                )
                diffs.append(
                    TokenDiff(
                        raw_index=r_idx,
                        corr_index=c_idx,
                        raw_token=raw_tok,
                        corr_token=corr_tok,
                        raw_start=raw_spans[r_idx][0],
                        raw_end=raw_spans[r_idx][1],
                        corr_start=corr_spans[c_idx][0],
                        corr_end=corr_spans[c_idx][1],
                        op="replace",
                        equal_ci=(raw_tok.lower() == corr_tok.lower()),
                        error_type=error_type,
                    )
                )
            for r_idx in range(i1 + width, i2):
                raw_tok = raw_tokens[r_idx]
                diffs.append(
                    TokenDiff(
                        raw_index=r_idx,
                        corr_index=None,
                        raw_token=raw_tok,
                        corr_token=None,
                        raw_start=raw_spans[r_idx][0],
                        raw_end=raw_spans[r_idx][1],
                        corr_start=None,
                        corr_end=None,
                        op="delete",
                        equal_ci=False,
                        error_type="PunctuationDeletion"
                        if not is_word(raw_tok)
                        else "Deletion",
                    )
                )
            for c_idx in range(j1 + width, j2):
                corr_tok = corr_tokens[c_idx]
                diffs.append(
                    TokenDiff(
                        raw_index=None,
                        corr_index=c_idx,
                        raw_token=None,
                        corr_token=corr_tok,
                        raw_start=None,
                        raw_end=None,
                        corr_start=corr_spans[c_idx][0],
                        corr_end=corr_spans[c_idx][1],
                        op="insert",
                        equal_ci=False,
                        error_type="PunctuationInsertion"
                        if not is_word(corr_tok)
                        else "Insertion",
                    )
                )
        elif tag == "delete":
            for r_idx in range(i1, i2):
                raw_tok = raw_tokens[r_idx]
                diffs.append(
                    TokenDiff(
                        raw_index=r_idx,
                        corr_index=None,
                        raw_token=raw_tok,
                        corr_token=None,
                        raw_start=raw_spans[r_idx][0],
                        raw_end=raw_spans[r_idx][1],
                        corr_start=None,
                        corr_end=None,
                        op="delete",
                        equal_ci=False,
                        error_type="PunctuationDeletion"
                        if not is_word(raw_tok)
                        else "Deletion",
                    )
                )
        elif tag == "insert":
            for c_idx in range(j1, j2):
                corr_tok = corr_tokens[c_idx]
                diffs.append(
                    TokenDiff(
                        raw_index=None,
                        corr_index=c_idx,
                        raw_token=None,
                        corr_token=corr_tok,
                        raw_start=None,
                        raw_end=None,
                        corr_start=corr_spans[c_idx][0],
                        corr_end=corr_spans[c_idx][1],
                        op="insert",
                        equal_ci=False,
                        error_type="PunctuationInsertion"
                        if not is_word(corr_tok)
                        else "Insertion",
                    )
                )
        else:  # pragma: no cover - exhaustiveness guard
            raise ValueError(f"Unexpected opcode {tag}")

    return diffs


# ---------------------------------------------------------------------------
# Word-map post processing
# ---------------------------------------------------------------------------


def _sentence_ids_from_tokens(tokens: Sequence[Optional[str]]) -> Dict[int, int]:
    sentence_ids: Dict[int, int] = {}
    sid = 0
    for idx, tok in enumerate(tokens):
        if tok is None:
            continue
        sentence_ids[idx] = sid
        if tok in TERMINALS:
            sid += 1
    return sentence_ids


def _first_alpha(text: str) -> Optional[str]:
    for ch in text:
        if ch.isalpha():
            return ch
    return None


def _boundary_labels(
    corr_index: Optional[int],
    corr_token: Optional[str],
    sentence_ids: Dict[int, int],
    sentence_starts: Dict[int, int],
    sentence_ends: Dict[int, int],
) -> Tuple[str, str, Optional[int]]:
    if corr_index is None:
        return "", "", None

    sid = sentence_ids.get(corr_index)
    if sid is None:
        return "", "", None

    parts: List[str] = []
    checks: List[str] = []

    if corr_index == sentence_starts.get(sid):
        parts.append("Sentence Beginning")
        first_alpha = _first_alpha(corr_token or "")
        if first_alpha is None:
            checks.append("Needs Review")
        elif first_alpha.isupper() or (corr_token in OPENING_PUNCT):
            checks.append("Correct Beginning")
        else:
            checks.append("Incorrect Beginning")

    if corr_index == sentence_ends.get(sid):
        parts.append("Sentence Ending")
        if corr_token in TERMINALS:
            checks.append("Correct Ending")
        else:
            checks.append("Incorrect Ending")

    return " | ".join(parts), " | ".join(checks), sid


def _prepare_sentence_metadata(diffs: Sequence[TokenDiff]) -> Dict[str, Dict[int, int]]:
    corr_tokens: Dict[int, Optional[str]] = {}
    for diff in diffs:
        if diff.corr_index is not None:
            corr_tokens[diff.corr_index] = diff.corr_token

    sentence_ids = _sentence_ids_from_tokens(
        [corr_tokens.get(i) for i in sorted(corr_tokens)]
    )
    # Map back to actual corr_index values
    sentence_ids = {
        corr_index: sentence_ids[idx]
        for idx, corr_index in enumerate(sorted(corr_tokens))
        if idx in sentence_ids
    }

    sentence_starts: Dict[int, int] = {}
    sentence_ends: Dict[int, int] = {}
    for corr_index, sid in sentence_ids.items():
        sentence_starts.setdefault(sid, corr_index)
        if corr_index not in sentence_ends or corr_index >= sentence_ends[sid]:
            sentence_ends[sid] = corr_index
    return {
        "sentence_ids": sentence_ids,
        "sentence_starts": sentence_starts,
        "sentence_ends": sentence_ends,
    }


def create_word_map_dataframe(
    df: pd.DataFrame,
    *,
    id_column: str = "ID",
    raw_column: str = "Raw text",
    corrected_column: str = "Corrected text",
) -> pd.DataFrame:
    """Construct the word-level alignment table."""

    if id_column not in df.columns:
        raise KeyError(f"Dataframe missing '{id_column}' column")
    if raw_column not in df.columns or corrected_column not in df.columns:
        raise KeyError("Raw and corrected text columns are required")

    rows: List[Dict[str, object]] = []
    global_row = 0

    for _, record in df.iterrows():
        doc_id = str(record[id_column])
        raw_text = str(record[raw_column])
        corr_text = str(record[corrected_column])

        diffs = build_word_map(raw_text, corr_text)
        if not diffs:
            diffs = [
                TokenDiff(
                    raw_index=None,
                    corr_index=None,
                    raw_token=None,
                    corr_token=None,
                    raw_start=None,
                    raw_end=None,
                    corr_start=None,
                    corr_end=None,
                    op="equal",
                    equal_ci=True,
                    error_type="Equal",
                )
            ]

        meta = _prepare_sentence_metadata(diffs)
        sentence_ids = meta["sentence_ids"]
        sentence_starts = meta["sentence_starts"]
        sentence_ends = meta["sentence_ends"]

        last_sid: Optional[int] = None
        for diff in diffs:
            boundary, check, sid = _boundary_labels(
                diff.corr_index, diff.corr_token, sentence_ids, sentence_starts, sentence_ends
            )
            if sid is None:
                sid = last_sid if last_sid is not None else 0
            last_sid = sid

            rows.append(
                {
                    "ID": doc_id,
                    "RowID": f"{doc_id}-{global_row:05d}",
                    "corr_index": diff.corr_index,
                    "raw_index": diff.raw_index,
                    "corr_token": diff.corr_token,
                    "raw_token": diff.raw_token,
                    "corr_start": diff.corr_start,
                    "corr_end": diff.corr_end,
                    "raw_start": diff.raw_start,
                    "raw_end": diff.raw_end,
                    "op": diff.op,
                    "equal_ci": diff.equal_ci,
                    "error_type": diff.error_type,
                    "Sentence Boundaries": boundary,
                    "BoundaryCheck": check,
                    "CorrSentenceID": sid,
                    "SentenceRef": f"S{sid + 1}",
                }
            )
            global_row += 1

    df_map = pd.DataFrame(rows)
    return df_map


# ---------------------------------------------------------------------------
# Sentence level summary (Step 9 equivalent)
# ---------------------------------------------------------------------------

NO_SPACE_BEFORE = set(".,;:!?)]}\"'»”’…")
NO_SPACE_AFTER = set("([{\"'«“‘")


def detokenize(tokens: Sequence[Optional[str]]) -> str:
    out: List[str] = []
    for token in tokens:
        if token is None or (isinstance(token, float) and math.isnan(token)):
            continue
        text = str(token)
        if not out:
            out.append(text)
            continue
        prev = out[-1]
        if text in NO_SPACE_BEFORE or re.fullmatch(r"[.]{3}", text):
            out[-1] = prev + text
        elif prev and prev[-1] == " ":
            out[-1] = prev + text
        elif prev in NO_SPACE_AFTER:
            out[-1] = prev + text
        else:
            out.append(" " + text)
    return "".join(out).strip()


def summarise_sentences(df_map: pd.DataFrame) -> pd.DataFrame:
    required = {
        "ID",
        "CorrSentenceID",
        "corr_token",
        "raw_token",
        "Sentence Boundaries",
        "BoundaryCheck",
        "SentenceRef",
    }
    missing = required - set(df_map.columns)
    if missing:
        raise KeyError(f"df_map missing required columns: {sorted(missing)}")

    sort_cols = ["ID", "CorrSentenceID"]
    if "corr_index" in df_map.columns:
        sort_cols.append("corr_index")
    ordered = df_map.sort_values(sort_cols, kind="mergesort")

    groups = ordered.groupby(["ID", "CorrSentenceID"], sort=False)

    rows: List[Dict[str, object]] = []
    for (doc_id, sid), group in groups:
        corr_tokens = group["corr_token"].tolist()
        raw_tokens = [tok for tok in group["raw_token"].tolist() if pd.notna(tok)]
        corr_sentence = detokenize(corr_tokens)
        raw_sentence = detokenize(raw_tokens)

        boundary_rows = group["Sentence Boundaries"].astype(str)
        begin_rows = group[boundary_rows.str.contains("Sentence Beginning", na=False)]
        end_rows = group[boundary_rows.str.contains("Sentence Ending", na=False)]

        def _check(rows: pd.Series, label: str) -> Optional[int]:
            if rows.empty:
                return None
            joined = " | ".join(group.loc[rows.index, "BoundaryCheck"].dropna().astype(str))
            if not joined:
                return None
            return 1 if label in joined else 0

        rows.append(
            {
                "ID": doc_id,
                "CorrSentenceID": sid,
                "SentenceRef": group["SentenceRef"].iloc[0],
                "CorrectedSentence": corr_sentence,
                "RawSentence": raw_sentence,
                "TokensInSentence": int(group.shape[0]),
                "EditsInSentence": int((group["op"] != "equal").sum())
                if "op" in group
                else np.nan,
                "Insertions": int((group["op"] == "insert").sum())
                if "op" in group
                else np.nan,
                "Deletions": int((group["op"] == "delete").sum())
                if "op" in group
                else np.nan,
                "Replacements": int((group["op"] == "replace").sum())
                if "op" in group
                else np.nan,
                "CorrectBeginning": _check(begin_rows, "Correct Beginning"),
                "CorrectEnding": _check(end_rows, "Correct Ending"),
                "HasHardTerminal": any(tok in TERMINALS for tok in corr_tokens),
                "HasOpeningQuote": any(tok in OPENING_PUNCT for tok in corr_tokens),
                "HasClosingQuote": any(tok in CLOSING_PUNCT for tok in corr_tokens),
                "CorrIndexMin": group["corr_index"].min()
                if "corr_index" in group
                else np.nan,
                "CorrIndexMax": group["corr_index"].max()
                if "corr_index" in group
                else np.nan,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _preview(df: pd.DataFrame, *, max_rows: int = 5) -> str:
    if df.empty:
        return "(empty dataframe)"
    return df.head(max_rows).to_string(index=False)


def run_pipeline(input_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_mechanical_data(input_csv)
    df = add_word_and_token_counts(df)
    df = ensure_corrected_column(df)
    df_map = create_word_map_dataframe(df)
    sentence_summary = summarise_sentences(df_map)
    return df, df_map, sentence_summary


def main(argv: Sequence[str]) -> int:
    if len(argv) < 2:
        print("Usage: python mechanical_pipeline.py <input_csv>")
        return 1
    input_csv = Path(argv[1])
    df, df_map, sentence_summary = run_pipeline(input_csv)

    print("\n=== Preprocessed Data (with counts) ===")
    print(_preview(df[["ID", "Raw text", "WordCount", "TokenCount", "Corrected text"]]))

    print("\n=== Word Map Extract ===")
    print(
        _preview(
            df_map[
                [
                    "ID",
                    "SentenceRef",
                    "corr_index",
                    "raw_token",
                    "corr_token",
                    "op",
                    "Sentence Boundaries",
                    "BoundaryCheck",
                ]
            ]
        )
    )

    print("\n=== Sentence Summary ===")
    print(
        _preview(
            sentence_summary[
                [
                    "ID",
                    "SentenceRef",
                    "CorrectedSentence",
                    "TokensInSentence",
                    "EditsInSentence",
                    "CorrectBeginning",
                    "CorrectEnding",
                ]
            ]
        )
    )

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI execution
    raise SystemExit(main(sys.argv))
