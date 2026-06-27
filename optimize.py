# ============================================================
#  optimize.py  —  Offline DSPy Prompt Optimization
#  Optimizer  : MIPROv2 (Bayesian, few-shot + instruction tuning)
#  Run this OFFLINE against your call transcripts CSV.
#  Output: optimized_pipeline.json  →  loaded by main.py at startup
#
#  Usage:
#    python optimize.py
#    python optimize.py --csv human_transcripts.csv --auto medium
# ============================================================

import os
import csv
import random
import argparse
import dspy
from dspy.teleprompt import MIPROv2
from dotenv import load_dotenv

# Import our pipeline definition (same module, no FastAPI overhead)
from main import SalesAdvisorPipeline

load_dotenv()

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
OPTIMIZED_WEIGHTS_PATH = "optimized_pipeline.json"

TRANSCRIPT_COLUMN_ALIASES = (
    "Call Transcript",
    "audio_transcript",
    "transcript",
    "Transcript",
)

BUY_LEAD_COLUMN_ALIASES = (
    "buylead",
    "buy_lead",
    "call_id",
)

VALID_ACTIONS = {
    "suggest_product",
    "handle_objection",
    "ask_clarification",
    "close_deal",
    "escalate_to_human",
    "end_call",
}

VALID_SENTIMENTS = {"positive", "neutral", "negative", "hesitant", "interested"}

VALID_OBJECTION_TYPES = {"price", "trust", "timing", "need", "competitor", "other", "none"}


# ─────────────────────────────────────────
# 1. CONFIGURE LM
# ─────────────────────────────────────────
lm = dspy.LM(
    model="gemini/gemini-2.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0.3,
)
dspy.configure(lm=lm)


# ─────────────────────────────────────────
# 2. LOAD & PARSE TRANSCRIPT CSV
#    Expected columns (at minimum):
#      - "Call Transcript"   : raw call text
#      - "buylead"           : "yes"/"no" — did call generate a buy lead?
#    Optional enrichment columns (used if present):
#      - "next_action"       : ground-truth action label
#      - "sentiment"         : ground-truth sentiment label
#      - "has_objection"     : "true"/"false"
# ─────────────────────────────────────────

def find_column(fieldnames: list[str] | None, candidates: tuple[str, ...]) -> str | None:
    if not fieldnames:
        return None

    normalized = {name.strip().lower(): name for name in fieldnames}
    for candidate in candidates:
        match = normalized.get(candidate.lower())
        if match:
            return match

    return None


def parse_buylead(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in ("", "0", "no", "false", "n"):
        return "no"
    return "yes"


def build_trainset(filepath: str, max_examples: int = 80) -> list[dspy.Example]:
    """
    Converts rows from human_transcripts.csv into dspy.Example objects.
    Each example represents one mid-call decision point.

    Strategy:
      - recent_transcript  = last ~800 chars of the transcript (simulates
                             what the ADE sees mid-call)
      - customer_state     = synthetic summary derived from buylead + row data
      - All other inputs   = sensible defaults so DSPy can run the full pipeline
      - Labels (outputs)   = derived from buylead when ground-truth cols absent
    """
    examples = []
    try:
        with open(filepath, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            transcript_column = find_column(reader.fieldnames, TRANSCRIPT_COLUMN_ALIASES)
            if not transcript_column:
                print(f"[ERROR] Could not find a transcript column in {filepath}.")
                print(f"        Expected one of: {', '.join(TRANSCRIPT_COLUMN_ALIASES)}")
                print(f"        Found columns: {reader.fieldnames}")
                return []

            buylead_column = find_column(reader.fieldnames, BUY_LEAD_COLUMN_ALIASES)
            if not buylead_column:
                print(f"[ERROR] Missing required label column in {filepath}.")
                print(f"        Expected one of: {', '.join(BUY_LEAD_COLUMN_ALIASES)}")
                print("        Values should be 'yes'/'no', or non-empty lead IDs for yes.")
                print(f"        Found columns: {reader.fieldnames}")
                return []

            if buylead_column != "buylead":
                print(f"[INFO] Using '{buylead_column}' as the buylead column.")

            rows = list(reader)
    except FileNotFoundError:
        print(f"[ERROR] {filepath} not found. Aborting.")
        return []

    random.shuffle(rows)
    rows = rows[:max_examples]

    for row in rows:
        transcript = row.get(transcript_column, "").strip()
        if not transcript:
            continue

        buylead = parse_buylead(row.get(buylead_column, ""))

        # ── Slice transcript to simulate "recent" turns ──
        recent = transcript[-800:] if len(transcript) > 800 else transcript

        # ── Derive labels from buylead when no explicit labels ──
        if buylead == "yes":
            next_action = row.get("next_action", "suggest_product").strip() or "suggest_product"
            sentiment   = row.get("sentiment",   "interested").strip()      or "interested"
            recommend   = True
        else:
            next_action = row.get("next_action", "ask_clarification").strip() or "ask_clarification"
            sentiment   = row.get("sentiment",   "hesitant").strip()          or "hesitant"
            recommend   = False

        # Validate labels against known enums
        if next_action not in VALID_ACTIONS:
            next_action = "ask_clarification"
        if sentiment not in VALID_SENTIMENTS:
            sentiment = "neutral"

        has_objection_raw = row.get("has_objection", "false").strip().lower()
        has_objection = has_objection_raw in ("true", "1", "yes")

        example = dspy.Example(
            # ── Inputs (pipeline.forward() args) ──
            recent_transcript=recent,
            customer_state=f"buylead={buylead}; budget=unknown; location=unknown",
            conversation_history="{}",
            available_inventory="No inventory data provided.",
            last_advisor_action="ask_clarification",

            # ── Expected outputs (used by metric) ──
            expected_next_action=next_action,
            expected_sentiment=sentiment,
            expected_has_objection=has_objection,
            expected_recommend_product=recommend,
        ).with_inputs(
            "recent_transcript",
            "customer_state",
            "conversation_history",
            "available_inventory",
            "last_advisor_action",
        )

        examples.append(example)

    print(f"[INFO] Built {len(examples)} training examples from {filepath}")
    return examples


# ─────────────────────────────────────────
# 3. METRIC FUNCTION
#    DSPy's optimizer maximizes this score.
#    We reward:
#      - Correct next_action        (most important — 0.5 weight)
#      - Correct sentiment          (0.2)
#      - Non-empty advisor_response (0.2) — proxy for "human-like output"
#      - Correct has_objection      (0.1)
#
#    WHY this metric for humanization?
#      next_action correctness is the primary proxy for "does the bot
#      understand the situation like a human advisor would?"
#      advisor_response non-empty check ensures the optimizer never
#      rewards the model for producing empty/refusal outputs.
# ─────────────────────────────────────────

def humanization_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    score = 0.0

    # 1. next_action match  (0.5)
    if getattr(pred, "next_action", "").strip().lower() == example.expected_next_action.lower():
        score += 0.5

    # 2. sentiment match  (0.2)
    if getattr(pred, "sentiment", "").strip().lower() == example.expected_sentiment.lower():
        score += 0.2

    # 3. advisor_response non-empty & reasonably long  (0.2)
    resp = getattr(pred, "advisor_response", "")
    if isinstance(resp, str) and len(resp.strip()) > 30:
        score += 0.2

    # 4. has_objection match  (0.1)
    pred_objection = bool(getattr(pred, "has_objection", False))
    if pred_objection == example.expected_has_objection:
        score += 0.1

    return score


# ─────────────────────────────────────────
# 4. OPTIMIZE
# ─────────────────────────────────────────

def run_optimization(csv_path: str, auto: str = "light"):
    """
    Runs MIPROv2 on the SalesAdvisorPipeline.

    MIPROv2 does TWO things that directly help with humanization:
      a) Bootstraps few-shot examples from your real transcripts and
         injects them into each predictor's prompt automatically.
      b) Proposes and tests multiple instruction variants per predictor,
         finding the phrasing that maximises humanization_metric.

    auto="light"  → ~6 trials, fast (~15–30 min on Gemini Flash)
    auto="medium" → ~12 trials, better quality (~45–90 min)
    auto="heavy"  → ~18 trials, best quality (~2–3 hrs)
    """
    examples = build_trainset(csv_path)
    if len(examples) < 10:
        print("[ERROR] Need at least 10 examples to optimize. Check your CSV.")
        return

    # Split 80/20 train/val
    split = int(len(examples) * 0.8)
    trainset = examples[:split]
    valset   = examples[split:]

    print(f"[INFO] Train: {len(trainset)}  Val: {len(valset)}")
    print(f"[INFO] Starting MIPROv2 optimization  (auto={auto}) ...")

    optimizer = MIPROv2(
        metric=humanization_metric,
        auto=auto,
        init_temperature=0.7,   # Higher = more diverse instruction proposals
        verbose=True,
    )

    baseline = SalesAdvisorPipeline()

    optimized = optimizer.compile(
        baseline.deepcopy(),
        trainset=trainset,
        valset=valset,
        max_bootstrapped_demos=3,   # Up to 3 real transcript snippets per predictor
        max_labeled_demos=4,        # Up to 4 labeled examples per predictor
    )

    optimized.save(OPTIMIZED_WEIGHTS_PATH)
    print(f"\n[SUCCESS] Optimized pipeline saved to: {OPTIMIZED_WEIGHTS_PATH}")
    print(f"          Restart main.py and it will auto-load the new weights.")

    # ── Evaluate before vs after ──
    evaluator = dspy.Evaluate(
        metric=humanization_metric,
        devset=valset,
        num_threads=4,
        display_progress=True,
        display_table=True,
    )

    print("\n── Baseline score ──")
    evaluator(SalesAdvisorPipeline())

    print("\n── Optimized score ──")
    evaluator(optimized)


# ─────────────────────────────────────────
# 5. CLI ENTRYPOINT
# ─────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize SalesAdvisorPipeline with MIPROv2")
    parser.add_argument(
        "--csv",
        default="human_transcripts.csv",
        help="Path to call transcripts CSV (default: human_transcripts.csv)"
    )
    parser.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        default="light",
        help="MIPROv2 optimization intensity (default: light)"
    )
    args = parser.parse_args()

    run_optimization(csv_path=args.csv, auto=args.auto)