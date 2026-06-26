# ============================================================
#  main.py  —  DSPy Sales Advisor API for Spinny Used Cars
#  LLM     : Gemini 2.5 Flash
#  Framework: DSPy + FastAPI
#  Language : Hindi / Hinglish transcripts
#
#  OPTIMIZATION FLOW:
#    Offline  →  run optimize.py  →  saves optimized_pipeline.json
#    Online   →  this file loads  optimized_pipeline.json at startup
#               and falls back to the unoptimized pipeline if not found
# ============================================================

import os
import csv
import random
import dspy
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────
# 1. CONFIGURE GEMINI 2.5 FLASH via DSPy
# ─────────────────────────────────────────
lm = dspy.LM(
    model="gemini/gemini-2.5-flash",
    api_key=os.environ.get("GEMINI_API_KEY"),
    temperature=0.3,
)
dspy.configure(lm=lm)

OPTIMIZED_WEIGHTS_PATH = "optimized_pipeline.json"


# ─────────────────────────────────────────
# 2. DSPy SIGNATURES
#    NOTE: few_shot_examples is now REMOVED
#    from all signatures. DSPy's MIPROv2
#    optimizer injects bootstrapped demos
#    directly into each predictor's prompt
#    — passing them as an input field
#    interferes with that mechanism.
# ─────────────────────────────────────────

class SentimentAnalysis(dspy.Signature):
    """
    Analyze customer sentiment from a Hindi/Hinglish sales call transcript.
    Classify as: positive, neutral, negative, hesitant, or interested.
    Focus on tone, word choice, hesitation patterns, and buying signals.
    """
    recent_transcript: str = dspy.InputField(
        desc="Recent conversation turns from the call (last 3–5 exchanges)"
    )
    customer_state: str = dspy.InputField(
        desc="Current structured state of the customer (budget, preferences, location)"
    )
    sentiment: str = dspy.OutputField(
        desc="One of: positive, neutral, negative, hesitant, interested"
    )
    sentiment_reason: str = dspy.OutputField(
        desc="Short reason explaining the sentiment in 1 sentence"
    )


class ObjectionHandler(dspy.Signature):
    """
    Detect if the customer has raised an objection or rebuttal in a
    Hindi/Hinglish used car sales call. If yes, generate a human-like
    advisor response that addresses it naturally — warm, not scripted.
    Match the register (Hinglish casual) of a real Spinny advisor.
    """
    recent_transcript: str = dspy.InputField(
        desc="Recent conversation turns"
    )
    customer_state: str = dspy.InputField(
        desc="Structured customer state"
    )
    has_objection: bool = dspy.OutputField(
        desc="True if customer raised an objection or rebuttal"
    )
    objection_type: str = dspy.OutputField(
        desc="Type: price, trust, timing, need, competitor, other, or none"
    )
    objection_response: str = dspy.OutputField(
        desc="Natural Hindi/Hinglish advisor response to handle the objection"
    )


class ProductRecommender(dspy.Signature):
    """
    Based on the customer's conversation and state, determine whether
    to recommend a used car or loan product and what to say.
    Only recommend when customer shows clear readiness signals.
    """
    recent_transcript: str = dspy.InputField(
        desc="Recent conversation turns"
    )
    customer_state: str = dspy.InputField(
        desc="Structured customer state including preferences"
    )
    available_inventory: str = dspy.InputField(
        desc="Summary of available used car inventory (if any)"
    )
    recommend_product: bool = dspy.OutputField(
        desc="True if customer is ready for a product/loan recommendation"
    )
    recommended_product: str = dspy.OutputField(
        desc="Specific car or loan product to recommend, or 'none'"
    )
    recommendation_pitch: str = dspy.OutputField(
        desc="Natural Hindi/Hinglish pitch line for the recommendation"
    )


class NextActionDecider(dspy.Signature):
    """
    Decide the next best action for a sales advisor in a Hindi/Hinglish
    used car call, based on full conversation context. The advisor_response
    must sound like a real human Spinny advisor — warm, conversational,
    using natural Hinglish fillers and not robotic phrasing.

    Next action must be one of:
    suggest_product, handle_objection, ask_clarification,
    close_deal, escalate_to_human, end_call.
    """
    recent_transcript: str = dspy.InputField(
        desc="Recent conversation turns"
    )
    customer_state: str = dspy.InputField(
        desc="Structured customer state"
    )
    conversation_history: str = dspy.InputField(
        desc="Prior structured conversation data"
    )
    sentiment: str = dspy.InputField(
        desc="Detected customer sentiment"
    )
    has_objection: bool = dspy.InputField(
        desc="Whether an objection was detected"
    )
    recommend_product: bool = dspy.InputField(
        desc="Whether a product recommendation is appropriate"
    )
    last_advisor_action: str = dspy.InputField(
        desc="Last action taken by the advisor"
    )
    next_action: str = dspy.OutputField(
        desc="One of: suggest_product, handle_objection, ask_clarification, close_deal, escalate_to_human, end_call"
    )
    advisor_response: str = dspy.OutputField(
        desc="Natural human-like Hindi/Hinglish response the advisor should say next"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief internal reasoning for this decision (not spoken aloud)"
    )


# ─────────────────────────────────────────
# 3. DSPy PIPELINE MODULE
# ─────────────────────────────────────────

class SalesAdvisorPipeline(dspy.Module):
    """
    Four-stage DSPy pipeline:
      1. Sentiment  →  how is the customer feeling?
      2. Objection  →  did they push back, and how to respond?
      3. Product    →  are they ready for a recommendation?
      4. Action     →  what should the advisor say/do next?

    When optimized via MIPROv2, each ChainOfThought predictor gets
    its own tuned instruction + bootstrapped few-shot demos baked in.
    The forward() method stays identical — only the internal prompts change.
    """

    def __init__(self):
        super().__init__()
        self.sentiment_analyzer  = dspy.ChainOfThought(SentimentAnalysis)
        self.objection_handler   = dspy.ChainOfThought(ObjectionHandler)
        self.product_recommender = dspy.ChainOfThought(ProductRecommender)
        self.next_action_decider = dspy.ChainOfThought(NextActionDecider)

    def forward(
        self,
        recent_transcript: str,
        customer_state: str,
        conversation_history: str,
        available_inventory: str,
        last_advisor_action: str,
    ):
        # Step 1: Sentiment
        sentiment_result = self.sentiment_analyzer(
            recent_transcript=recent_transcript,
            customer_state=customer_state,
        )

        # Step 2: Objection detection + response
        objection_result = self.objection_handler(
            recent_transcript=recent_transcript,
            customer_state=customer_state,
        )

        # Step 3: Product recommendation readiness
        product_result = self.product_recommender(
            recent_transcript=recent_transcript,
            customer_state=customer_state,
            available_inventory=available_inventory,
        )

        # Step 4: Next action decision using all signals
        action_result = self.next_action_decider(
            recent_transcript=recent_transcript,
            customer_state=customer_state,
            conversation_history=conversation_history,
            sentiment=sentiment_result.sentiment,
            has_objection=objection_result.has_objection,
            recommend_product=product_result.recommend_product,
            last_advisor_action=last_advisor_action,
        )

        return dspy.Prediction(
            sentiment=sentiment_result.sentiment,
            sentiment_reason=sentiment_result.sentiment_reason,
            has_objection=objection_result.has_objection,
            objection_type=objection_result.objection_type,
            objection_response=objection_result.objection_response,
            recommend_product=product_result.recommend_product,
            recommended_product=product_result.recommended_product,
            recommendation_pitch=product_result.recommendation_pitch,
            next_action=action_result.next_action,
            advisor_response=action_result.advisor_response,
            reasoning=action_result.reasoning,
        )


# ─────────────────────────────────────────
# 4. LOAD PIPELINE (optimized or baseline)
# ─────────────────────────────────────────

pipeline = SalesAdvisorPipeline()

if os.path.exists(OPTIMIZED_WEIGHTS_PATH):
    pipeline.load(OPTIMIZED_WEIGHTS_PATH)
    print(f"[INFO] Loaded optimized pipeline from {OPTIMIZED_WEIGHTS_PATH}")
else:
    print(f"[INFO] No optimized weights found — using baseline pipeline.")
    print(f"       Run `python optimize.py` to generate {OPTIMIZED_WEIGHTS_PATH}")


# ─────────────────────────────────────────
# 5. FASTAPI APP
# ─────────────────────────────────────────

app = FastAPI(
    title="Spinny Sales Advisor DSPy API",
    description="AI sales advisor for used car loan calls in Hindi/Hinglish",
    version="2.0.0"
)


# ── Request / Response Models ──────────────

class AdvisorRequest(BaseModel):
    recent_transcript: List[Any]
    customer_state: Dict[str, Any]
    conversation_history: Dict[str, Any]
    available_inventory: Optional[Dict[str, Any]] = None
    last_advisor_action: Optional[str] = "none"


class AdvisorResponse(BaseModel):
    next_action: str
    advisor_response: str
    sentiment: str
    sentiment_reason: str
    has_objection: bool
    objection_type: str
    objection_response: str
    recommend_product: bool
    recommended_product: str
    recommendation_pitch: str
    reasoning: str


# ── Auth Helper ────────────────────────────

def verify_token(authorization: Optional[str]):
    expected = os.environ.get("API_SECRET_TOKEN")
    if expected and authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="Unauthorized")


# ── Main Endpoint ──────────────────────────

@app.post("/dspy/advise", response_model=AdvisorResponse)
async def advise(
    payload: AdvisorRequest,
    authorization: Optional[str] = Header(None)
):
    verify_token(authorization)

    result = pipeline(
        recent_transcript=str(payload.recent_transcript),
        customer_state=str(payload.customer_state),
        conversation_history=str(payload.conversation_history),
        available_inventory=str(payload.available_inventory)
                           if payload.available_inventory else "No inventory data provided.",
        last_advisor_action=payload.last_advisor_action or "none",
    )

    return AdvisorResponse(
        next_action=result.next_action,
        advisor_response=result.advisor_response,
        sentiment=result.sentiment,
        sentiment_reason=result.sentiment_reason,
        has_objection=result.has_objection,
        objection_type=result.objection_type,
        objection_response=result.objection_response,
        recommend_product=result.recommend_product,
        recommended_product=result.recommended_product,
        recommendation_pitch=result.recommendation_pitch,
        reasoning=result.reasoning,
    )


# ── Health Check ───────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "optimized": os.path.exists(OPTIMIZED_WEIGHTS_PATH),
        "model": "gemini-2.5-flash",
        "version": "2.0.0",
    }


# ── Local Dev ──────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)