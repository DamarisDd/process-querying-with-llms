from __future__ import annotations
from collections import Counter

import json
import csv
import re
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple


# =========================
# Configuration
# =========================

INPUT_JSON_PATH = "evaluation_input.json"
OUTPUT_DETAILS_CSV = "evaluation_details.csv"
OUTPUT_SUMMARY_CSV = "evaluation_summary.csv"
OUTPUT_JSON_REPORT = "evaluation_report.json"

IGNORE_CASE = True
STRIP_SPACES = True
COLLAPSE_INTERNAL_SPACES = True
REMOVE_DUPLICATES_FOR_METRICS = False
PRINT_DETAILED_RESULTS = True
REPLACE_ARROW_WITH_COMMA = True
NORMALIZE_COMMA_SPACING = True

EXPECTED_CATEGORIES = ["SPARQL-generating agent", "ChatGPT Retrieval Plugin-based (RAG) agent"]
EXPECTED_LEVELS = ["Level 1", "Level 2", "Level 3"]
EXPECTED_RUNS_PER_QUESTION = 3


# =========================
# Normalization and whatnot
# =========================

def normalize_item(
    value: Any,
    ignore_case: bool = True,
    strip_spaces: bool = True,
    collapse_internal_spaces: bool = True,
    replace_arrow_with_comma: bool = True,
    normalize_comma_spacing: bool = True
) -> str:
    if value is None:
        text = ""
    else:
        text = str(value)

    if replace_arrow_with_comma:
        for old in ["->", "→", "⇒", "⟶", "➝", "→"]:
            text = text.replace(old, ",")

    if normalize_comma_spacing:
        text = re.sub(r"\s*,\s*", ", ", text)

    if strip_spaces:
        text = text.strip()

    if collapse_internal_spaces:
        text = " ".join(text.split())

    if ignore_case:
        text = text.lower()

    return text


def normalize_list(
    values: List[Any],
    ignore_case: bool = True,
    strip_spaces: bool = True,
    collapse_internal_spaces: bool = True
) -> List[str]:
    return [
        normalize_item(
            value=v,
            ignore_case=ignore_case,
            strip_spaces=strip_spaces,
            collapse_internal_spaces=collapse_internal_spaces,
            replace_arrow_with_comma=REPLACE_ARROW_WITH_COMMA,
            normalize_comma_spacing=NORMALIZE_COMMA_SPACING
        )
        for v in values
    ]


def unique_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in values:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result



# =========================
# Metric computation
# =========================

def compute_set_metrics(
    predicted: List[Any],
    ground_truth: List[Any],
    ignore_case: bool = True,
    strip_spaces: bool = True,
    collapse_internal_spaces: bool = True,
    remove_duplicates_for_metrics: bool = True
) -> Dict[str, Any]:
    if predicted is None:
        raise ValueError("Predicted list is None.")
    if ground_truth is None:
        raise ValueError("Ground-truth list is None.")

    predicted_norm = normalize_list(
        predicted, ignore_case, strip_spaces, collapse_internal_spaces
    )
    ground_truth_norm = normalize_list(
        ground_truth, ignore_case, strip_spaces, collapse_internal_spaces
    )

    if remove_duplicates_for_metrics:
        predicted_eval = unique_preserve_order(predicted_norm)
        ground_truth_eval = unique_preserve_order(ground_truth_norm)

        predicted_counter = Counter(predicted_eval)
        ground_truth_counter = Counter(ground_truth_eval)
    else:
        predicted_eval = predicted_norm
        ground_truth_eval = ground_truth_norm

        predicted_counter = Counter(predicted_eval)
        ground_truth_counter = Counter(ground_truth_eval)

    matched_counter = predicted_counter & ground_truth_counter
    extra_counter = predicted_counter - ground_truth_counter
    missing_counter = ground_truth_counter - predicted_counter

    tp = sum(matched_counter.values())
    fp = sum(extra_counter.values())
    fn = sum(missing_counter.values())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    matched_items = sorted(matched_counter.elements())
    missing_items = sorted(missing_counter.elements())
    extra_items = sorted(extra_counter.elements())

    return {
        "predicted_normalized": predicted_eval,
        "ground_truth_normalized": ground_truth_eval,
        "matched_items": matched_items,
        "missing_items": missing_items,
        "extra_items": extra_items,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def compute_order_metrics(
    predicted: List[Any],
    ground_truth: List[Any],
    ignore_case: bool = True,
    strip_spaces: bool = True,
    collapse_internal_spaces: bool = True
) -> Dict[str, Any]:
    predicted_norm = normalize_list(
        predicted, ignore_case, strip_spaces, collapse_internal_spaces
    )
    ground_truth_norm = normalize_list(
        ground_truth, ignore_case, strip_spaces, collapse_internal_spaces
    )

    order_exact_match = predicted_norm == ground_truth_norm
    same_items_ignoring_order = set(predicted_norm) == set(ground_truth_norm)

    return {
        "order_exact_match": order_exact_match,
        "same_items_ignoring_order": same_items_ignoring_order
    }


def evaluate_single_prediction(
    predicted: List[Any],
    ground_truth: List[Any]
) -> Dict[str, Any]:
    set_metrics = compute_set_metrics(
        predicted=predicted,
        ground_truth=ground_truth,
        ignore_case=IGNORE_CASE,
        strip_spaces=STRIP_SPACES,
        collapse_internal_spaces=COLLAPSE_INTERNAL_SPACES,
        remove_duplicates_for_metrics=REMOVE_DUPLICATES_FOR_METRICS
    )

    order_metrics = compute_order_metrics(
        predicted=predicted,
        ground_truth=ground_truth,
        ignore_case=IGNORE_CASE,
        strip_spaces=STRIP_SPACES,
        collapse_internal_spaces=COLLAPSE_INTERNAL_SPACES
    )

    fully_correct = (
        set_metrics["precision"] == 1.0
        and set_metrics["recall"] == 1.0
        and order_metrics["order_exact_match"]
    )

    return {
        **set_metrics,
        **order_metrics,
        "fully_correct": fully_correct
    }


# =========================
# Aggregation
# =========================

def safe_mean(values: List[float]) -> float:
    return mean(values) if values else 0.0


def aggregate_results(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "n_predictions": 0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
            "mean_f1": 0.0,
            "fully_correct_rate": 0.0,
            "order_exact_match_rate": 0.0,
            "same_items_ignoring_order_rate": 0.0
        }

    return {
        "n_predictions": len(rows),
        "mean_precision": safe_mean([row["precision"] for row in rows]),
        "mean_recall": safe_mean([row["recall"] for row in rows]),
        "mean_f1": safe_mean([row["f1"] for row in rows]),
        "fully_correct_rate": safe_mean([1.0 if row["fully_correct"] else 0.0 for row in rows]),
        "order_exact_match_rate": safe_mean([1.0 if row["order_exact_match"] else 0.0 for row in rows]),
        "same_items_ignoring_order_rate": safe_mean(
            [1.0 if row["same_items_ignoring_order"] else 0.0 for row in rows]
        )
    }


# =========================
# I/O
# =========================

def load_input_data(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_input_data(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_details_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)

    fieldnames = [
        "category",
        "level",
        "question_id",
        "question_text",
        "run_index",
        "ground_truth_raw",
        "predicted_raw",
        "ground_truth_normalized",
        "predicted_normalized",
        "matched_items",
        "missing_items",
        "extra_items",
        "precision",
        "recall",
        "f1",
        "same_items_ignoring_order",
        "order_exact_match",
        "fully_correct"
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "category": row["category"],
                "level": row["level"],
                "question_id": row["question_id"],
                "question_text": row["question_text"],
                "run_index": row["run_index"],
                "ground_truth_raw": json.dumps(row["ground_truth_raw"], ensure_ascii=False),
                "predicted_raw": json.dumps(row["predicted_raw"], ensure_ascii=False),
                "ground_truth_normalized": json.dumps(row["ground_truth_normalized"], ensure_ascii=False),
                "predicted_normalized": json.dumps(row["predicted_normalized"], ensure_ascii=False),
                "matched_items": json.dumps(row["matched_items"], ensure_ascii=False),
                "missing_items": json.dumps(row["missing_items"], ensure_ascii=False),
                "extra_items": json.dumps(row["extra_items"], ensure_ascii=False),
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "same_items_ignoring_order": row["same_items_ignoring_order"],
                "order_exact_match": row["order_exact_match"],
                "fully_correct": row["fully_correct"]
            })


def save_summary_csv(summary_rows: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)

    fieldnames = [
        "scope_type",
        "scope_name",
        "category",
        "level",
        "n_predictions",
        "mean_precision",
        "mean_recall",
        "mean_f1",
        "fully_correct_rate",
        "order_exact_match_rate",
        "same_items_ignoring_order_rate"
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def save_json_report(report: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


# =========================
# Succint reports
# =========================

def print_detailed_results(rows: List[Dict[str, Any]]) -> None:
    for row in rows:
        print("-" * 90)
        print(
            f"Details for category='{row['category']}', "
            f"level='{row['level']}', "
            f"question_id='{row['question_id']}', "
            f"predicted_list={row['run_index']}"
        )
        print("-" * 90)
        print(f"Question:          {row['question_text']}")
        print(f"Ground truth used: {row['ground_truth_raw']}")
        print(f"Predicted list:    {row['predicted_raw']}")
        print(f"Matched items:     {row['matched_items']}")
        print(f"Missing items:     {row['missing_items']}")
        print(f"Extra items:       {row['extra_items']}")
        print(f"Precision:         {row['precision']:.2%}")
        print(f"Recall:            {row['recall']:.2%}")
        print(f"F1-score:          {row['f1']:.2%}")
        print(f"Same items:        {row['same_items_ignoring_order']}")
        print(f"Order exact match: {row['order_exact_match']}")
        print(f"Fully correct:     {row['fully_correct']}")
        print()


def print_summary(summary_by_scope: Dict[str, Dict[str, Any]]) -> None:
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    for scope, metrics in summary_by_scope.items():
        print(scope)
        print(f"  Number of predictions:        {metrics['n_predictions']}")
        print(f"  Mean precision:               {metrics['mean_precision']:.2%}")
        print(f"  Mean recall:                  {metrics['mean_recall']:.2%}")
        print(f"  Mean F1:                      {metrics['mean_f1']:.2%}")
        print(f"  Fully correct rate:           {metrics['fully_correct_rate']:.2%}")
        print(f"  Order exact match rate:       {metrics['order_exact_match_rate']:.2%}")
        print(f"  Same-items-ignore-order rate: {metrics['same_items_ignoring_order_rate']:.2%}")
        print()


# =========================
# Possible encountered errors
# =========================

def validate_input_structure(data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("Top-level JSON structure must be a dictionary.")

    for category, levels_dict in data.items():
        if not isinstance(levels_dict, dict):
            raise ValueError(
                f"Category '{category}' must map to a dictionary of levels."
            )

        for level, questions in levels_dict.items():
            if not isinstance(questions, list):
                raise ValueError(
                    f"Level '{level}' in category '{category}' must map to a list of question entries."
                )

            for idx, question_entry in enumerate(questions, start=1):
                if not isinstance(question_entry, dict):
                    raise ValueError(
                        f"Question entry #{idx} in category '{category}', level '{level}' must be a dictionary."
                    )

                if "ground_truth" not in question_entry:
                    raise ValueError(
                        f"Missing 'ground_truth' in category '{category}', level '{level}', question #{idx}."
                    )

                if "predicted_lists" not in question_entry:
                    raise ValueError(
                        f"Missing 'predicted_lists' in category '{category}', level '{level}', question #{idx}."
                    )

                predicted_lists = question_entry["predicted_lists"]
                if not isinstance(predicted_lists, list):
                    raise ValueError(
                        f"'predicted_lists' must be a list in category '{category}', level '{level}', question #{idx}."
                    )


# =========================
# Main eval
# =========================

def evaluate_dataset(
    data: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    detailed_rows: List[Dict[str, Any]] = []

    for category_name, levels_dict in data.items():
        for level_name, questions in levels_dict.items():
            for question_index, item in enumerate(questions, start=1):
                question_id = item.get("question_id", f"{category_name}-{level_name}-Q{question_index}")
                question_text = item.get("question_text", "")
                ground_truth = item.get("ground_truth", [])
                predicted_lists = item.get("predicted_lists", [])

                if len(predicted_lists) != EXPECTED_RUNS_PER_QUESTION:
                    print(
                        f"Warning: question '{question_id}' in category '{category_name}', "
                        f"level '{level_name}' has {len(predicted_lists)} predicted lists "
                        f"instead of {EXPECTED_RUNS_PER_QUESTION}."
                    )

                for run_index, predicted in enumerate(predicted_lists, start=1):
                    if predicted is None:
                        predicted = []

                    result = evaluate_single_prediction(
                        predicted=predicted,
                        ground_truth=ground_truth
                    )

                    detailed_rows.append({
                        "category": category_name,
                        "level": level_name,
                        "question_id": question_id,
                        "question_text": question_text,
                        "run_index": run_index,
                        "ground_truth_raw": ground_truth,
                        "predicted_raw": predicted,
                        "ground_truth_normalized": result["ground_truth_normalized"],
                        "predicted_normalized": result["predicted_normalized"],
                        "matched_items": result["matched_items"],
                        "missing_items": result["missing_items"],
                        "extra_items": result["extra_items"],
                        "precision": result["precision"],
                        "recall": result["recall"],
                        "f1": result["f1"],
                        "same_items_ignoring_order": result["same_items_ignoring_order"],
                        "order_exact_match": result["order_exact_match"],
                        "fully_correct": result["fully_correct"]
                    })

    summary_by_scope: Dict[str, Dict[str, Any]] = {}

    categories = sorted(set(row["category"] for row in detailed_rows))
    category_level_pairs = sorted(set((row["category"], row["level"]) for row in detailed_rows))

    for category in categories:
        category_rows = [row for row in detailed_rows if row["category"] == category]
        summary_by_scope[f"CATEGORY :: {category}"] = aggregate_results(category_rows)

    for category, level in category_level_pairs:
        level_rows = [
            row for row in detailed_rows
            if row["category"] == category and row["level"] == level
        ]
        summary_by_scope[f"LEVEL :: {category} :: {level}"] = aggregate_results(level_rows)

    summary_by_scope["OVERALL"] = aggregate_results(detailed_rows)

    summary_rows_for_csv: List[Dict[str, Any]] = []

    for category in categories:
        metrics = summary_by_scope[f"CATEGORY :: {category}"]
        summary_rows_for_csv.append({
            "scope_type": "category",
            "scope_name": category,
            "category": category,
            "level": "",
            **metrics
        })

    for category, level in category_level_pairs:
        metrics = summary_by_scope[f"LEVEL :: {category} :: {level}"]
        summary_rows_for_csv.append({
            "scope_type": "level",
            "scope_name": f"{category} :: {level}",
            "category": category,
            "level": level,
            **metrics
        })

    overall_metrics = summary_by_scope["OVERALL"]
    summary_rows_for_csv.append({
        "scope_type": "overall",
        "scope_name": "OVERALL",
        "category": "",
        "level": "",
        **overall_metrics
    })

    return detailed_rows, summary_by_scope, summary_rows_for_csv


def main() -> None:
    data = load_input_data(INPUT_JSON_PATH)

    def canonical_qid(qid: str) -> str:
        qid = str(qid).strip()

        # If it's already in canonical form, keep it.
        if re.fullmatch(r"L[123]-Q\d+", qid):
            return qid

        # If it has a prefix like S-L3-Q1 or R-L2-Q7, extract the canonical tail.
        match = re.search(r"(L[123]-Q\d+)$", qid)
        if match:
            return match.group(1)

        # Otherwise leave it unchanged.
        return qid

    replacement_texts_level2 = {
        "L2-Q1": "If the complaint is admissible, what is the maximum total execution time until the complaint is closed? To answer the question, identify the condition or source element and the exact target element, find all valid paths, including boundary-event continuations where relevant, that satisfy them, apply the ad-hoc subprocess assumptions, compute or extract the requested metrics for each path, treat handoffs, where relevant, as transfers of work between tasks or subprocesses assigned to different responsible individuals on the evaluated path, compare the paths and return the result from the best valid one.",
        "L2-Q2": "If the complaint is admissible, which tasks belong to the path with the maximum total execution time until the complaint is closed? To answer the question, identify the condition or source element and the exact target element, find all valid paths, including boundary-event continuations where relevant, that satisfy them, apply the ad-hoc subprocess assumptions, compute or extract the requested metrics for each path, treat handoffs, where relevant, as transfers of work between tasks or subprocesses assigned to different responsible individuals on the evaluated path, compare the paths and return the result from the best valid one.",
        "L2-Q3": "If the complaint is admissible, what is the minimum total execution time until the complaint is closed? To answer the question, identify the condition or source element and the exact target element, find all valid paths, including boundary-event continuations where relevant, that satisfy them, apply the ad-hoc subprocess assumptions, compute or extract the requested metrics for each path, treat handoffs, where relevant, as transfers of work between tasks or subprocesses assigned to different responsible individuals on the evaluated path, compare the paths and return the result from the best valid one.",
        "L2-Q4": "If the complaint is admissible, which tasks belong to the path with the minimum total execution time until the complaint is closed? To answer the question, identify the condition or source element and the exact target element, find all valid paths, including boundary-event continuations where relevant, that satisfy them, apply the ad-hoc subprocess assumptions, compute or extract the requested metrics for each path, treat handoffs, where relevant, as transfers of work between tasks or subprocesses assigned to different responsible individuals on the evaluated path, compare the paths and return the result from the best valid one.",
        "L2-Q5": "What is the maximum total execution time starting from \"Identify required documents and records\" until \"Complaint closed\"? To answer the question, identify the condition or source element and the exact target element, find all valid paths, including boundary-event continuations where relevant, that satisfy them, apply the ad-hoc subprocess assumptions, compute or extract the requested metrics for each path, treat handoffs, where relevant, as transfers of work between tasks or subprocesses assigned to different responsible individuals on the evaluated path, compare the paths and return the result from the best valid one.",
        "L2-Q6": "Which tasks belong to the path with the maximum total execution time starting from \"Identify required documents and records\" until \"Complaint closed\"? To answer the question, identify the condition or source element and the exact target element, find all valid paths, including boundary-event continuations where relevant, that satisfy them, apply the ad-hoc subprocess assumptions, compute or extract the requested metrics for each path, treat handoffs, where relevant, as transfers of work between tasks or subprocesses assigned to different responsible individuals on the evaluated path, compare the paths and return the result from the best valid one.",
        "L2-Q7": "What is the minimum total execution time starting from \"Identify required documents and records\" until \"Complaint closed\"? To answer the question, identify the condition or source element and the exact target element, find all valid paths, including boundary-event continuations where relevant, that satisfy them, apply the ad-hoc subprocess assumptions, compute or extract the requested metrics for each path, treat handoffs, where relevant, as transfers of work between tasks or subprocesses assigned to different responsible individuals on the evaluated path, compare the paths and return the result from the best valid one.",
        "L2-Q8": "Which tasks belong to the path with the minimum total execution time starting from \"Identify required documents and records\" until \"Complaint closed\"? To answer the question, identify the condition or source element and the exact target element, find all valid paths, including boundary-event continuations where relevant, that satisfy them, apply the ad-hoc subprocess assumptions, compute or extract the requested metrics for each path, treat handoffs, where relevant, as transfers of work between tasks or subprocesses assigned to different responsible individuals on the evaluated path, compare the paths and return the result from the best valid one.",
        "L2-Q9": "If the complaint is admissible, what is the maximum total cost until the complaint is closed? To answer the question, identify the condition or source element and the exact target element, find all valid paths, including boundary-event continuations where relevant, that satisfy them, apply the ad-hoc subprocess assumptions, compute or extract the requested metrics for each path, treat handoffs, where relevant, as transfers of work between tasks or subprocesses assigned to different responsible individuals on the evaluated path, compare the paths and return the result from the best valid one.",
        "L2-Q10": "If the complaint is admissible, which tasks belong to the path with the maximum total cost until the complaint is closed? To answer the question, identify the condition or source element and the exact target element, find all valid paths, including boundary-event continuations where relevant, that satisfy them, apply the ad-hoc subprocess assumptions, compute or extract the requested metrics for each path, treat handoffs, where relevant, as transfers of work between tasks or subprocesses assigned to different responsible individuals on the evaluated path, compare the paths and return the result from the best valid one.",
        "L2-Q11": "If the complaint is admissible, what is the minimum total cost until the complaint is closed? To answer the question, identify the condition or source element and the exact target element, find all valid paths, including boundary-event continuations where relevant, that satisfy them, apply the ad-hoc subprocess assumptions, compute or extract the requested metrics for each path, treat handoffs, where relevant, as transfers of work between tasks or subprocesses assigned to different responsible individuals on the evaluated path, compare the paths and return the result from the best valid one.",
        "L2-Q12": "If the complaint is admissible, which tasks belong to the path with the minimum total cost until the complaint is closed? To answer the question, identify the condition or source element and the exact target element, find all valid paths, including boundary-event continuations where relevant, that satisfy them, apply the ad-hoc subprocess assumptions, compute or extract the requested metrics for each path, treat handoffs, where relevant, as transfers of work between tasks or subprocesses assigned to different responsible individuals on the evaluated path, compare the paths and return the result from the best valid one.",
        "L2-Q13": "What is the maximum total cost starting from \"Identify required documents and records\" until \"Complaint closed\"? To answer the question, identify the condition or source element and the exact target element, find all valid paths, including boundary-event continuations where relevant, that satisfy them, apply the ad-hoc subprocess assumptions, compute or extract the requested metrics for each path, treat handoffs, where relevant, as transfers of work between tasks or subprocesses assigned to different responsible individuals on the evaluated path, compare the paths and return the result from the best valid one.",
        "L2-Q14": "Which tasks belong to the path with the maximum total cost starting from \"Identify required documents and records\" until \"Complaint closed\"? To answer the question, identify the condition or source element and the exact target element, find all valid paths, including boundary-event continuations where relevant, that satisfy them, apply the ad-hoc subprocess assumptions, compute or extract the requested metrics for each path, treat handoffs, where relevant, as transfers of work between tasks or subprocesses assigned to different responsible individuals on the evaluated path, compare the paths and return the result from the best valid one.",
        "L2-Q15": "What is the minimum total cost starting from \"Identify required documents and records\" until \"Complaint closed\"? To answer the question, identify the condition or source element and the exact target element, find all valid paths, including boundary-event continuations where relevant, that satisfy them, apply the ad-hoc subprocess assumptions, compute or extract the requested metrics for each path, treat handoffs, where relevant, as transfers of work between tasks or subprocesses assigned to different responsible individuals on the evaluated path, compare the paths and return the result from the best valid one.",
        "L2-Q16": "Which tasks belong to the path with the minimum total cost starting from \"Identify required documents and records\" until \"Complaint closed\"? To answer the question, identify the condition or source element and the exact target element, find all valid paths, including boundary-event continuations where relevant, that satisfy them, apply the ad-hoc subprocess assumptions, compute or extract the requested metrics for each path, treat handoffs, where relevant, as transfers of work between tasks or subprocesses assigned to different responsible individuals on the evaluated path, compare the paths and return the result from the best valid one.",
        "L2-Q17": "If the complaint is admissible and until the complaint is closed, which tasks belong to the path that involves the highest number of distinct responsible individuals? To answer the question, identify the condition or source element and the exact target element, find all valid paths, including boundary-event continuations where relevant, that satisfy them, apply the ad-hoc subprocess assumptions, compute or extract the requested metrics for each path, treat handoffs, where relevant, as transfers of work between tasks or subprocesses assigned to different responsible individuals on the evaluated path, compare the paths and return the result from the best valid one.",
        "L2-Q18": "If the complaint is admissible, which responsible individual appears most often on the path until the complaint is closed? To answer the question, identify the condition or source element and the exact target element, find all valid paths, including boundary-event continuations where relevant, that satisfy them, apply the ad-hoc subprocess assumptions, compute or extract the requested metrics for each path, treat handoffs, where relevant, as transfers of work between tasks or subprocesses assigned to different responsible individuals on the evaluated path, compare the paths and return the result from the best valid one.",
        "L2-Q19": "If the complaint is admissible, which handoffs occur on the execution path with the highest number of responsibility handoffs until the complaint is closed? To answer the question, identify the condition or source element and the exact target element, find all valid paths, including boundary-event continuations where relevant, that satisfy them, apply the ad-hoc subprocess assumptions, compute or extract the requested metrics for each path, treat handoffs, where relevant, as transfers of work between tasks or subprocesses assigned to different responsible individuals on the evaluated path, compare the paths and return the result from the best valid one."
    }
    
    level_name = "Level 2"
    categories_to_update = ["SPARQL-generating agent", "ChatGPT Retrieval Plugin-based (RAG) agent"]

    for category in categories_to_update:
        for item in data.get(category, {}).get(level_name, []):
            qid = item.get("question_id", "")
            canonical_id = canonical_qid(qid)

            if canonical_id in replacement_texts_level2:
                item["question_text"] = replacement_texts_level2[canonical_id]

    COMMON_L3_SUFFIX = """ To answer the question:
    - Identify the source, the stated condition, if applicable, and the exact target; if no explicit source is given, begin from the start of the relevant process or subprocess.
    - Keep the exact target fixed throughout the reasoning.
    - Treat the condition as satisfied once the path passes through the required gateway outcome; after that, any later continuation, including interrupting or non-interrupting boundary-event continuations, remains eligible if it can still reach the exact target.
    - From the selected branch onward, trace every reachable continuation through tasks, subprocesses, nested subprocesses and boundary-event continuations.
    - Keep a candidate path only if it ends at the exact target named in the question.
    - For ad-hoc subprocesses, do not require internal sequence flows to prove internal reachability.
    - In ad-hoc subprocesses, do not require internal sequence flows to prove internal reachability; internal tasks or subprocesses may be skipped or occur at most once, while modeled dependencies must still be respected.
    - If a boundary event is attached to an ad-hoc subprocess, it may occur after one or more compatible internal tasks or subprocesses have completed.
    - Use query methods to inspect sequence flows, boundary events, subprocess contents and target reachability before answering.
    - For each valid target-reaching path, compute or extract the requested result.
    - Compare only the valid target-reaching paths and return the result from the best one."""

    replacement_texts_level3 = {
        "L3-Q1": f"""If the complaint is admissible, what is the maximum total execution time until the complaint is closed?{COMMON_L3_SUFFIX}""",
        "L3-Q2": f"""If the complaint is admissible, which tasks belong to the path with the maximum total execution time until the complaint is closed?{COMMON_L3_SUFFIX}""",
        "L3-Q3": f"""If the complaint is admissible, what is the minimum total execution time until the complaint is closed?{COMMON_L3_SUFFIX}""",
        "L3-Q4": f"""If the complaint is admissible, which tasks belong to the path with the minimum total execution time until the complaint is closed?{COMMON_L3_SUFFIX}""",
        "L3-Q5": f"""What is the maximum total execution time starting from "Identify required documents and records" until "Complaint closed"?{COMMON_L3_SUFFIX}""",
        "L3-Q6": f"""Which tasks belong to the path with the maximum total execution time starting from "Identify required documents and records" until "Complaint closed"?{COMMON_L3_SUFFIX}""",
        "L3-Q7": f"""What is the minimum total execution time starting from "Identify required documents and records" until "Complaint closed"?{COMMON_L3_SUFFIX}""",
        "L3-Q8": f"""Which tasks belong to the path with the minimum total execution time starting from "Identify required documents and records" until "Complaint closed"?{COMMON_L3_SUFFIX}""",
        "L3-Q9": f"""If the complaint is admissible, what is the maximum total cost until the complaint is closed?{COMMON_L3_SUFFIX}""",
        "L3-Q10": f"""If the complaint is admissible, which tasks belong to the path with the maximum total cost until the complaint is closed?{COMMON_L3_SUFFIX}""",
        "L3-Q11": f"""If the complaint is admissible, what is the minimum total cost until the complaint is closed?{COMMON_L3_SUFFIX}""",
        "L3-Q12": f"""If the complaint is admissible, which tasks belong to the path with the minimum total cost until the complaint is closed?{COMMON_L3_SUFFIX}""",
        "L3-Q13": f"""What is the maximum total cost starting from "Identify required documents and records" until "Complaint closed"?{COMMON_L3_SUFFIX}""",
        "L3-Q14": f"""Which tasks belong to the path with the maximum total cost starting from "Identify required documents and records" until "Complaint closed"?{COMMON_L3_SUFFIX}""",
        "L3-Q15": f"""What is the minimum total cost starting from "Identify required documents and records" until "Complaint closed"?{COMMON_L3_SUFFIX}""",
        "L3-Q16": f"""Which tasks belong to the path with the minimum total cost starting from "Identify required documents and records" until "Complaint closed"?{COMMON_L3_SUFFIX}""",
        "L3-Q17": f"""If the complaint is admissible and until the complaint is closed, which tasks belong to the path that involves the highest number of distinct responsible individuals?{COMMON_L3_SUFFIX}""",
        "L3-Q18": f"""If the complaint is admissible, which responsible individual appears most often on the path until the complaint is closed?{COMMON_L3_SUFFIX}""",
        "L3-Q19": f"""If the complaint is admissible, which handoffs occur on the execution path with the highest number of responsibility handoffs until the complaint is closed?{COMMON_L3_SUFFIX}"""
    }

    level3_updates = 0

    level_name = "Level 3"
    categories_to_update = ["SPARQL-generating agent", "ChatGPT Retrieval Plugin-based (RAG) agent"]

    for category in categories_to_update:
        for item in data.get(category, {}).get(level_name, []):
            qid = item.get("question_id", "")
            canonical_id = canonical_qid(qid)

            if canonical_id in replacement_texts_level3:
                item["question_text"] = replacement_texts_level3[canonical_id]
                level3_updates += 1
            else:
                print(f"[L3 no match] category={category} raw_qid={qid!r} canonical={canonical_id!r}")

    print(f"Level 3 question_text updates applied: {level3_updates}")
    
    save_input_data(INPUT_JSON_PATH, data)

    validate_input_structure(data)

    detailed_rows, summary_by_scope, summary_rows_for_csv = evaluate_dataset(data)

    if PRINT_DETAILED_RESULTS:
        print_detailed_results(detailed_rows)

    print_summary(summary_by_scope)

    save_details_csv(detailed_rows, OUTPUT_DETAILS_CSV)
    save_summary_csv(summary_rows_for_csv, OUTPUT_SUMMARY_CSV)

    json_report = {
        "config": {
            "input_json_path": INPUT_JSON_PATH,
            "ignore_case": IGNORE_CASE,
            "strip_spaces": STRIP_SPACES,
            "collapse_internal_spaces": COLLAPSE_INTERNAL_SPACES,
            "remove_duplicates_for_metrics": REMOVE_DUPLICATES_FOR_METRICS,
            "expected_categories": EXPECTED_CATEGORIES,
            "expected_levels": EXPECTED_LEVELS,
            "expected_runs_per_question": EXPECTED_RUNS_PER_QUESTION
        },
        "details": detailed_rows,
        "summary": summary_by_scope
    }
    save_json_report(json_report, OUTPUT_JSON_REPORT)

    print("Files written successfully:")
    print(f"  - {OUTPUT_DETAILS_CSV}")
    print(f"  - {OUTPUT_SUMMARY_CSV}")
    print(f"  - {OUTPUT_JSON_REPORT}")


if __name__ == "__main__":
    main()