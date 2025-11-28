import os
import io
import json
from typing import Dict, Any, Optional

import httpx
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


AIPIPE_API_KEY = os.getenv("AIPIPE_API_KEY", "")
QUIZ_SECRET = os.getenv("QUIZ_SECRET", "")


AIPIPE_CHAT_URL = "https://aipipe.org/openai/v1/chat/completions"

DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-nano")


class QuizRequest(BaseModel):
    secret: str
    question: str
    csv_url: Optional[str] = None
    csv_text: Optional[str] = None


class QuizResponse(BaseModel):
    answer: str
    score: float
    checks: Dict[str, Any]
    stats: Dict[str, Any]
    quiz_facts: Dict[str, Any]



def call_llm_via_aipipe(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Call AIPipe's OpenAI-compatible /chat/completions endpoint
    and return the assistant's message content.
    """
    api_key = AIPIPE_API_KEY
    if not api_key:
        raise RuntimeError("AIPIPE_API_KEY environment variable is not set.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful data analyst."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 900,
    }

    with httpx.Client(timeout=90) as client:
        resp = client.post(AIPIPE_CHAT_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    return data["choices"][0]["message"]["content"]



def compute_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute basic stats for the dataset."""
    stats: Dict[str, Any] = {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "columns": list(df.columns),
    }

    numeric = df.select_dtypes(include="number")
    numeric_summary: Dict[str, Any] = {}
    for col in numeric.columns:
        series = numeric[col].dropna()
        if series.empty:
            continue
        numeric_summary[col] = {
            "mean": float(series.mean()),
            "min": float(series.min()),
            "max": float(series.max()),
            "std": float(series.std()),
            "na_count": int(df[col].isna().sum()),
        }
    stats["numeric_summary"] = numeric_summary

    na_counts = df.isna().sum().to_dict()
    stats["missing_by_column"] = {k: int(v) for k, v in na_counts.items()}

    return stats


def compute_quiz_facts(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute 'ground truth' facts from a video game sales dataset
    that we expect the LLM to mention in its analysis.
    Designed for columns like:
    Name, Platform, Year, Genre, Publisher,
    NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales
    """
    facts: Dict[str, Any] = {}

    facts["total_games"] = int(len(df))
    if "Platform" in df.columns:
        facts["n_platforms"] = int(df["Platform"].nunique())
    else:
        facts["n_platforms"] = None

    if "Genre" in df.columns:
        facts["n_genres"] = int(df["Genre"].nunique())
    else:
        facts["n_genres"] = None

    if "Publisher" in df.columns:
        facts["n_publishers"] = int(df["Publisher"].nunique())
    else:
        facts["n_publishers"] = None

    for col in ["NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales", "Global_Sales"]:
        if col in df.columns:
            facts[f"total_{col}"] = float(df[col].sum())
        else:
            facts[f"total_{col}"] = None

    if "Global_Sales" in df.columns and "Name" in df.columns:
        idx = df["Global_Sales"].idxmax()
        top_row = df.loc[idx]
        facts["top_game_name"] = str(top_row.get("Name", ""))
        facts["top_game_platform"] = str(top_row.get("Platform", ""))
        facts["top_game_genre"] = str(top_row.get("Genre", ""))
        facts["top_game_global_sales"] = float(top_row.get("Global_Sales", 0.0))
    else:
        facts["top_game_name"] = ""
        facts["top_game_platform"] = ""
        facts["top_game_genre"] = ""
        facts["top_game_global_sales"] = None

    if "Genre" in df.columns and not df["Genre"].dropna().empty:
        facts["top_genre_by_count"] = df["Genre"].mode()[0]
        genre_sales = (
            df.groupby("Genre")["Global_Sales"].sum()
            if "Global_Sales" in df.columns
            else None
        )
        if genre_sales is not None and not genre_sales.empty:
            facts["top_genre_by_sales"] = str(genre_sales.idxmax())
        else:
            facts["top_genre_by_sales"] = None
    else:
        facts["top_genre_by_count"] = None
        facts["top_genre_by_sales"] = None

    if "Platform" in df.columns and not df["Platform"].dropna().empty:
        facts["top_platform_by_count"] = str(df["Platform"].value_counts().idxmax())
        platform_sales = (
            df.groupby("Platform")["Global_Sales"].sum()
            if "Global_Sales" in df.columns
            else None
        )
        if platform_sales is not None and not platform_sales.empty:
            facts["top_platform_by_sales"] = str(platform_sales.idxmax())
        else:
            facts["top_platform_by_sales"] = None
    else:
        facts["top_platform_by_count"] = None
        facts["top_platform_by_sales"] = None

    if "Publisher" in df.columns and not df["Publisher"].dropna().empty:
        facts["top_publisher_by_count"] = str(df["Publisher"].value_counts().idxmax())
    else:
        facts["top_publisher_by_count"] = None

    if "Year" in df.columns:
        year_series = df["Year"].dropna()
        try:
            year_series = year_series.astype(int)
        except Exception:
            pass
        if not year_series.empty:
            facts["year_most_releases"] = int(year_series.value_counts().idxmax())
        else:
            facts["year_most_releases"] = None
    else:
        facts["year_most_releases"] = None

    region_totals = {}
    if facts.get("total_NA_Sales") is not None:
        region_totals["NA"] = facts["total_NA_Sales"]
    if facts.get("total_EU_Sales") is not None:
        region_totals["EU"] = facts["total_EU_Sales"]
    if facts.get("total_JP_Sales") is not None:
        region_totals["JP"] = facts["total_JP_Sales"]
    if facts.get("total_Other_Sales") is not None:
        region_totals["Other"] = facts["total_Other_Sales"]

    if region_totals:
        facts["top_region_by_sales"] = max(region_totals, key=region_totals.get)
    else:
        facts["top_region_by_sales"] = None

    return facts


def build_prompt(question: str, stats: Dict[str, Any], sample_rows: pd.DataFrame) -> str:
    """
    Build a prompt that describes the dataset and the quiz question,
    asking the LLM for a structured analysis + answer.
    """
    system_instructions = (
        "You are a data analyst. You are given a summary of a video game sales "
        "dataset and a small sample of rows. A question is provided that you "
        "must answer using the data. "
        "Write a clear analysis with these sections:\n"
        "1. Answer to the question, clearly stated.\n"
        "2. High-level summary of the dataset (1–2 short paragraphs).\n"
        "3. 5 numbered insights or patterns you notice.\n"
        "4. Possible data-quality issues (missing data, outliers, etc.).\n"
        "5. 3 suggested next steps or further analyses.\n\n"
        "Use numbered lists for sections 3–5."
    )

    user_payload = {
        "question": question,
        "dataset_summary": stats,
        "sample_rows": sample_rows.to_dict(orient="records"),
    }

    prompt = (
        f"SYSTEM INSTRUCTIONS:\n{system_instructions}\n\n"
        "USER DATA (JSON):\n"
        f"{json.dumps(user_payload, indent=2)}\n\n"
        "Now answer the question and provide the requested analysis."
    )

    return prompt


def grade_llm_answer(llm_text: str, stats: Dict[str, Any], facts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Grade the LLM's free-form analysis using:
    - generic checks (structure, row/col counts, etc.)
    - dataset-specific checks (top game, top genre, platform, publisher, etc.)

    Returns a dict with score_out_of_10 and checks.
    """
    text = llm_text.lower()
    score = 0.0
    checks: Dict[str, Any] = {}

    def contains(sub: Optional[str]) -> bool:
        return bool(sub) and sub.lower() in text

    checks["mentions_row_count"] = str(stats["rows"]) in text
    if checks["mentions_row_count"]:
        score += 0.5

    checks["mentions_col_count"] = str(stats["cols"]) in text
    if checks["mentions_col_count"]:
        score += 0.5

    any_col_mentioned = any(
        isinstance(col, str) and col and col.lower() in text
        for col in stats["columns"]
    )
    checks["mentions_any_column_name"] = any_col_mentioned
    if any_col_mentioned:
        score += 0.5

    checks["uses_numbered_list"] = ("1." in llm_text) and ("2." in llm_text)
    if checks["uses_numbered_list"]:
        score += 0.5

    checks["mentions_top_game_name"] = contains(facts.get("top_game_name"))
    if checks["mentions_top_game_name"]:
        score += 2.0

    checks["mentions_top_genre"] = contains(facts.get("top_genre_by_count"))
    if checks["mentions_top_genre"]:
        score += 1.5

    checks["mentions_top_platform_by_count"] = contains(facts.get("top_platform_by_count"))
    checks["mentions_top_platform_by_sales"] = contains(facts.get("top_platform_by_sales"))
    if checks["mentions_top_platform_by_count"] or checks["mentions_top_platform_by_sales"]:
        score += 1.5

    checks["mentions_top_publisher"] = contains(facts.get("top_publisher_by_count"))
    if checks["mentions_top_publisher"]:
        score += 1.0

    year = facts.get("year_most_releases")
    if year is not None:
        checks["mentions_year_most_releases"] = str(year) in text
    else:
        checks["mentions_year_most_releases"] = False
    if checks["mentions_year_most_releases"]:
        score += 1.0

    top_region = facts.get("top_region_by_sales")
    checks["mentions_top_region_by_sales"] = contains(top_region)
    if checks["mentions_top_region_by_sales"]:
        score += 1.0

    val = facts.get("top_game_global_sales")
    approx_ok = False
    if isinstance(val, (int, float)):
        approx_strings = {str(round(val, 1)), str(round(val))}
        approx_ok = any(s in text for s in approx_strings)
    checks["mentions_top_game_sales_value"] = approx_ok
    if approx_ok:
        score += 1.0

    total_games = facts.get("total_games")
    checks["mentions_total_games"] = total_games is not None and str(total_games) in text
    if checks["mentions_total_games"]:
        score += 1.0

    final_score = round(min(score, 10.0), 2)
    checks["raw_score"] = score
    return {"score_out_of_10": final_score, "checks": checks}


def load_dataset_from_request(req: QuizRequest) -> pd.DataFrame:
    if req.csv_url:
        with httpx.Client(timeout=60) as client:
            r = client.get(req.csv_url)
            r.raise_for_status()
            content = r.text
        df = pd.read_csv(io.StringIO(content))
        return df
    elif req.csv_text:
        df = pd.read_csv(io.StringIO(req.csv_text))
        return df
    else:
        raise HTTPException(status_code=400, detail="No CSV data provided (csv_url or csv_text required).")


app = FastAPI(title="LLM Analysis Quiz Solver")


@app.get("/")
def root():
    return {
        "message": "LLM Analysis Quiz API. POST to /solve with a quiz task.",
        "status": "ok",
    }


@app.post("/solve", response_model=QuizResponse)
def solve_quiz(req: QuizRequest):
    if not QUIZ_SECRET:
        raise HTTPException(status_code=500, detail="Server QUIZ_SECRET is not configured.")
    if req.secret != QUIZ_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret.")

    try:
        df = load_dataset_from_request(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading dataset: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="Dataset is empty.")

    stats = compute_stats(df)
    quiz_facts = compute_quiz_facts(df)

    prompt = build_prompt(req.question, stats, df.head(5))

    try:
        llm_text = call_llm_via_aipipe(prompt)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calling LLM: {e}")

    grading = grade_llm_answer(llm_text, stats, quiz_facts)

    return QuizResponse(
        answer=llm_text,
        score=grading["score_out_of_10"],
        checks=grading["checks"],
        stats=stats,
        quiz_facts=quiz_facts,
    )
