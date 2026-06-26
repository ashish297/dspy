import os

import dspy


OPTIMIZED_WEIGHTS_PATH = "optimized_pipeline.json"


class SentimentAnalysis(dspy.Signature):
    """
    Analyze customer sentiment from a Hindi/Hinglish sales call transcript.
    Classify as: positive, neutral, negative, hesitant, or interested.
    Focus on tone, word choice, hesitation patterns, and buying signals.
    """
    recent_transcript: str = dspy.InputField(
        desc="Recent conversation turns from the call (last 3-5 exchanges)"
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
    advisor response that addresses it naturally - warm, not scripted.
    Match the register (Hinglish casual) of a real Spinny advisor.
    """
    recent_transcript: str = dspy.InputField(desc="Recent conversation turns")
    customer_state: str = dspy.InputField(desc="Structured customer state")
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
    recent_transcript: str = dspy.InputField(desc="Recent conversation turns")
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
    must sound like a real human Spinny advisor - warm, conversational,
    using natural Hinglish fillers and not robotic phrasing.
    """
    recent_transcript: str = dspy.InputField(desc="Recent conversation turns")
    customer_state: str = dspy.InputField(desc="Structured customer state")
    conversation_history: str = dspy.InputField(desc="Prior structured conversation data")
    sentiment: str = dspy.InputField(desc="Detected customer sentiment")
    has_objection: bool = dspy.InputField(desc="Whether an objection was detected")
    recommend_product: bool = dspy.InputField(
        desc="Whether a product recommendation is appropriate"
    )
    last_advisor_action: str = dspy.InputField(desc="Last action taken by the advisor")
    next_action: str = dspy.OutputField(
        desc="One of: suggest_product, handle_objection, ask_clarification, close_deal, escalate_to_human, end_call"
    )
    advisor_response: str = dspy.OutputField(
        desc="Natural human-like Hindi/Hinglish response the advisor should say next"
    )
    reasoning: str = dspy.OutputField(
        desc="Brief internal reasoning for this decision (not spoken aloud)"
    )


class SalesAdvisorPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.sentiment_analyzer = dspy.ChainOfThought(SentimentAnalysis)
        self.objection_handler = dspy.ChainOfThought(ObjectionHandler)
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
        sentiment_result = self.sentiment_analyzer(
            recent_transcript=recent_transcript,
            customer_state=customer_state,
        )

        objection_result = self.objection_handler(
            recent_transcript=recent_transcript,
            customer_state=customer_state,
        )

        product_result = self.product_recommender(
            recent_transcript=recent_transcript,
            customer_state=customer_state,
            available_inventory=available_inventory,
        )

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


def build_lm(gemini_api_key: str, gemini_model: str) -> dspy.LM:
    return dspy.LM(
        model=gemini_model,
        api_key=gemini_api_key,
        temperature=0.3,
    )


def build_pipeline() -> SalesAdvisorPipeline:
    advisor_pipeline = SalesAdvisorPipeline()
    if os.path.exists(OPTIMIZED_WEIGHTS_PATH):
        advisor_pipeline.load(OPTIMIZED_WEIGHTS_PATH)
    return advisor_pipeline
