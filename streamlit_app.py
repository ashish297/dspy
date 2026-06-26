import json
import os
from typing import Any

import streamlit as st
from dotenv import load_dotenv


load_dotenv()


DEFAULT_TRANSCRIPT = [
    {"speaker": "advisor", "text": "Namaste sir, aap used car dekh rahe the?"},
    {"speaker": "customer", "text": "Haan, but price thoda zyada lag raha hai."},
]

DEFAULT_CUSTOMER_STATE = {
    "budget": "6 lakh",
    "location": "Delhi",
    "preference": "automatic hatchback",
}

DEFAULT_CONVERSATION_HISTORY = {
    "lead_source": "website",
    "previous_interest": "Hyundai i20",
}

DEFAULT_CARS = [
    {
        "model": "Hyundai i20",
        "year": 2021,
        "price": "5.8 lakh",
        "transmission": "automatic",
    }
]

ACTION_OPTIONS = [
    "none",
    "ask_clarification",
    "handle_objection",
    "suggest_product",
    "close_deal",
    "escalate_to_human",
    "end_call",
]

GEMINI_MODEL_OPTIONS = [
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-pro",
    "gemini/gemini-1.5-flash",
    "gemini/gemini-1.5-pro",
    "custom",
]


def initialize_state() -> None:
    if "transcript" not in st.session_state:
        st.session_state.transcript = DEFAULT_TRANSCRIPT.copy()
    if "cars" not in st.session_state:
        st.session_state.cars = DEFAULT_CARS.copy()
    if "advisor_response_payload" not in st.session_state:
        st.session_state.advisor_response_payload = None
    if "advisor_error" not in st.session_state:
        st.session_state.advisor_error = None
    if "advisor_run_count" not in st.session_state:
        st.session_state.advisor_run_count = 0
    if "last_request_payload" not in st.session_state:
        st.session_state.last_request_payload = None


def parse_extra_json(raw_json: str, label: str) -> dict[str, Any]:
    raw_json = raw_json.strip()
    if not raw_json:
        return {}

    try:
        parsed = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        st.error(f"{label} must be valid JSON: {exc.msg}")
        st.stop()

    if not isinstance(parsed, dict):
        st.error(f"{label} must be a JSON object.")
        st.stop()

    return parsed


def build_payload(
    customer_state: dict[str, Any],
    conversation_history: dict[str, Any],
    include_inventory: bool,
    last_advisor_action: str,
) -> dict[str, Any]:
    transcript = [
        {"speaker": turn["speaker"], "text": turn["text"].strip()}
        for turn in st.session_state.transcript
        if turn["text"].strip()
    ]

    cars = [
        {
            "model": car["model"].strip(),
            "year": car["year"],
            "price": car["price"].strip(),
            "transmission": car["transmission"].strip(),
        }
        for car in st.session_state.cars
        if car["model"].strip()
    ]

    return {
        "recent_transcript": transcript,
        "customer_state": customer_state,
        "conversation_history": conversation_history,
        "available_inventory": {"cars": cars} if include_inventory else None,
        "last_advisor_action": last_advisor_action,
    }


def get_secret_value(key: str) -> str:
    try:
        value = st.secrets.get(key, "")
    except Exception:
        value = ""

    return str(value) if value else ""


def get_default_api_key() -> str:
    load_dotenv()

    return os.environ.get("GEMINI_API_KEY") or get_secret_value("GEMINI_API_KEY")


def get_default_model() -> str:
    return os.environ.get("GEMINI_MODEL") or get_secret_value("GEMINI_MODEL") or GEMINI_MODEL_OPTIONS[0]


def configure_runtime(gemini_api_key: str) -> None:
    if gemini_api_key:
        os.environ["GEMINI_API_KEY"] = gemini_api_key


def redact_secret(text: str, secret: str) -> str:
    if secret:
        return text.replace(secret, "[redacted]")

    return text


def format_runtime_error(exc: Exception, gemini_api_key: str) -> str:
    message = redact_secret(str(exc), gemini_api_key)
    if "403" in message or "Forbidden" in message:
        return (
            "Gemini returned 403 Forbidden. Check that the API key is valid, "
            "the Gemini API is enabled for its Google project, the key is not "
            "blocked by API restrictions, and the selected model is available for that key.\n\n"
            f"Raw error: {message}"
        )

    if "socksio" in message.lower():
        return "SOCKS proxy support is missing. Run `pip install socksio` or reinstall from requirements.txt."

    return message


def test_gemini_connection(gemini_api_key: str, gemini_model: str) -> str:
    import litellm

    response = litellm.completion(
        model=gemini_model,
        api_key=gemini_api_key,
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
        max_tokens=5,
    )
    return response.choices[0].message.content or ""


@st.cache_resource(show_spinner="Loading DSPy advisor pipeline...")
def load_pipeline(gemini_api_key: str, gemini_model: str):
    configure_runtime(gemini_api_key)

    import dspy
    from advisor_core import build_lm, build_pipeline

    lm = build_lm(gemini_api_key, gemini_model)
    advisor_pipeline = build_pipeline()

    return advisor_pipeline, lm


def run_advisor(payload: dict[str, Any], gemini_api_key: str, gemini_model: str) -> dict[str, Any]:
    import dspy

    advisor_pipeline, lm = load_pipeline(gemini_api_key, gemini_model)
    with dspy.context(lm=lm):
        result = advisor_pipeline(
            recent_transcript=str(payload["recent_transcript"]),
            customer_state=str(payload["customer_state"]),
            conversation_history=str(payload["conversation_history"]),
            available_inventory=str(payload["available_inventory"])
            if payload["available_inventory"]
            else "No inventory data provided.",
            last_advisor_action=payload["last_advisor_action"] or "none",
        )

    return {
        "next_action": result.next_action,
        "advisor_response": result.advisor_response,
        "sentiment": result.sentiment,
        "sentiment_reason": result.sentiment_reason,
        "has_objection": result.has_objection,
        "objection_type": result.objection_type,
        "objection_response": result.objection_response,
        "recommend_product": result.recommend_product,
        "recommended_product": result.recommended_product,
        "recommendation_pitch": result.recommendation_pitch,
        "reasoning": result.reasoning,
    }


def render_status_chip(label: str, value: Any) -> None:
    st.metric(label, str(value))


def render_response(response_data: dict[str, Any]) -> None:
    st.subheader("Advisor Output")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_status_chip("Next Action", response_data.get("next_action", "-"))
    with col2:
        render_status_chip("Sentiment", response_data.get("sentiment", "-"))
    with col3:
        render_status_chip("Objection", response_data.get("has_objection", "-"))
    with col4:
        render_status_chip("Recommend", response_data.get("recommend_product", "-"))

    st.markdown("#### Advisor Response")
    st.info(response_data.get("advisor_response", "No advisor response returned."))

    st.markdown("#### Sentiment Reason")
    st.write(response_data.get("sentiment_reason", "-"))

    if response_data.get("has_objection"):
        st.markdown("#### Objection Handling")
        st.write(f"Type: **{response_data.get('objection_type', '-')}**")
        st.success(response_data.get("objection_response", "-"))

    if response_data.get("recommend_product"):
        st.markdown("#### Product Recommendation")
        st.write(f"Recommended product: **{response_data.get('recommended_product', '-')}**")
        st.success(response_data.get("recommendation_pitch", "-"))

    with st.expander("Internal reasoning"):
        st.write(response_data.get("reasoning", "-"))

    with st.expander("Raw API response"):
        st.json(response_data)


st.set_page_config(page_title="DSPy Sales Advisor", layout="wide")
initialize_state()

st.title("DSPy Sales Advisor")
st.caption("Build a request payload, run the DSPy advisor pipeline, and inspect the response.")

with st.sidebar:
    st.header("Runtime")
    st.write("Mode: single Streamlit app")
    stored_api_key = get_default_api_key()
    entered_api_key = st.text_input(
        "Gemini API key",
        value="",
        type="password",
        key="gemini_api_key_input",
        placeholder="Uses GEMINI_API_KEY from secrets if left blank",
        help="Paste a Gemini API key here to override GEMINI_API_KEY from .env or Streamlit secrets.",
    )
    gemini_api_key = entered_api_key.strip() or stored_api_key
    default_model = get_default_model()
    default_model_index = (
        GEMINI_MODEL_OPTIONS.index(default_model)
        if default_model in GEMINI_MODEL_OPTIONS
        else GEMINI_MODEL_OPTIONS.index("custom")
    )
    selected_model = st.selectbox("Gemini model", GEMINI_MODEL_OPTIONS, index=default_model_index)
    custom_model = st.text_input(
        "Custom model",
        value=default_model if selected_model == "custom" else "",
        placeholder="gemini/gemini-2.5-flash",
        disabled=selected_model != "custom",
    )
    gemini_model = custom_model.strip() if selected_model == "custom" else selected_model
    st.write("GEMINI_API_KEY:", "configured" if gemini_api_key.strip() else "missing")
    st.write("Active model:", gemini_model or "missing")
    if st.button("Clear model cache", use_container_width=True):
        load_pipeline.clear()
        st.success("Model cache cleared. The next run will reload the pipeline.")
    if st.button("Test Gemini connection", use_container_width=True):
        if not gemini_api_key.strip():
            st.error("Add a Gemini API key before testing the connection.")
        elif not gemini_model.strip():
            st.error("Choose or enter a Gemini model before testing the connection.")
        else:
            with st.spinner("Testing Gemini connection..."):
                try:
                    test_response = test_gemini_connection(gemini_api_key.strip(), gemini_model.strip())
                except Exception as exc:
                    st.error(format_runtime_error(exc, gemini_api_key.strip()))
                else:
                    st.success(f"Gemini connection succeeded: {test_response.strip()}")
    st.caption("For Streamlit Cloud, you can paste the key here or add GEMINI_API_KEY in app secrets.")

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    st.subheader("Request Inputs")

    st.markdown("#### Recent Transcript")
    transcript_to_remove = None
    for index, turn in enumerate(st.session_state.transcript):
        cols = st.columns([0.2, 0.65, 0.15])
        with cols[0]:
            turn["speaker"] = st.selectbox(
                "Speaker",
                ["advisor", "customer"],
                index=["advisor", "customer"].index(turn.get("speaker", "customer")),
                key=f"speaker_{index}",
            )
        with cols[1]:
            turn["text"] = st.text_area("Text", value=turn.get("text", ""), key=f"text_{index}", height=80)
        with cols[2]:
            st.write("")
            st.write("")
            if st.button("Remove", key=f"remove_turn_{index}"):
                transcript_to_remove = index

    if transcript_to_remove is not None:
        st.session_state.transcript.pop(transcript_to_remove)
        st.rerun()

    if st.button("Add transcript turn"):
        st.session_state.transcript.append({"speaker": "customer", "text": ""})
        st.rerun()

    st.markdown("#### Customer State")
    customer_col1, customer_col2, customer_col3 = st.columns(3)
    with customer_col1:
        budget = st.text_input("Budget", value=DEFAULT_CUSTOMER_STATE["budget"])
    with customer_col2:
        location = st.text_input("Location", value=DEFAULT_CUSTOMER_STATE["location"])
    with customer_col3:
        preference = st.text_input("Preference", value=DEFAULT_CUSTOMER_STATE["preference"])
    customer_extra = st.text_area("Extra customer_state JSON", value="{}", height=90)

    st.markdown("#### Conversation History")
    history_col1, history_col2 = st.columns(2)
    with history_col1:
        lead_source = st.text_input("Lead source", value=DEFAULT_CONVERSATION_HISTORY["lead_source"])
    with history_col2:
        previous_interest = st.text_input("Previous interest", value=DEFAULT_CONVERSATION_HISTORY["previous_interest"])
    history_extra = st.text_area("Extra conversation_history JSON", value="{}", height=90)

    st.markdown("#### Available Inventory")
    include_inventory = st.checkbox("Include inventory", value=True)
    car_to_remove = None
    if include_inventory:
        for index, car in enumerate(st.session_state.cars):
            cols = st.columns([0.3, 0.16, 0.22, 0.22, 0.1])
            with cols[0]:
                car["model"] = st.text_input("Model", value=car.get("model", ""), key=f"model_{index}")
            with cols[1]:
                car["year"] = st.number_input(
                    "Year",
                    min_value=1990,
                    max_value=2035,
                    value=int(car.get("year", 2021)),
                    key=f"year_{index}",
                )
            with cols[2]:
                car["price"] = st.text_input("Price", value=car.get("price", ""), key=f"price_{index}")
            with cols[3]:
                car["transmission"] = st.text_input(
                    "Transmission",
                    value=car.get("transmission", "automatic"),
                    key=f"transmission_{index}",
                )
            with cols[4]:
                st.write("")
                st.write("")
                if st.button("Remove", key=f"remove_car_{index}"):
                    car_to_remove = index

        if car_to_remove is not None:
            st.session_state.cars.pop(car_to_remove)
            st.rerun()

        if st.button("Add inventory car"):
            st.session_state.cars.append({"model": "", "year": 2021, "price": "", "transmission": "automatic"})
            st.rerun()

    st.markdown("#### Advisor State")
    last_advisor_action = st.selectbox(
        "Last advisor action",
        ACTION_OPTIONS,
        index=ACTION_OPTIONS.index("ask_clarification"),
    )

customer_state = {
    "budget": budget,
    "location": location,
    "preference": preference,
    **parse_extra_json(customer_extra, "Extra customer_state JSON"),
}
conversation_history = {
    "lead_source": lead_source,
    "previous_interest": previous_interest,
    **parse_extra_json(history_extra, "Extra conversation_history JSON"),
}
payload = build_payload(customer_state, conversation_history, include_inventory, last_advisor_action)

with right:
    st.subheader("Generated Payload")
    st.json(payload)

    run_col, clear_col = st.columns([0.7, 0.3])
    with run_col:
        submitted = st.button("Run Advisor", type="primary", use_container_width=True)
    with clear_col:
        clear_output = st.button("Clear", use_container_width=True)

    if clear_output:
        st.session_state.advisor_response_payload = None
        st.session_state.advisor_error = None
        st.session_state.last_request_payload = None
        st.rerun()

    if submitted:
        st.session_state.advisor_response_payload = None
        st.session_state.advisor_error = None
        st.session_state.last_request_payload = payload
        st.session_state.advisor_run_count += 1

        if not payload["recent_transcript"]:
            st.session_state.advisor_error = "Add at least one transcript turn before running the advisor."
        elif not gemini_api_key.strip():
            st.session_state.advisor_error = "Add a Gemini API key in the sidebar before running the advisor."
        elif not gemini_model.strip():
            st.session_state.advisor_error = "Choose or enter a Gemini model in the sidebar before running the advisor."
        else:
            with st.spinner(f"Running advisor pipeline... run #{st.session_state.advisor_run_count}"):
                try:
                    response_payload = run_advisor(payload, gemini_api_key.strip(), gemini_model.strip())
                except Exception as exc:
                    st.session_state.advisor_error = format_runtime_error(exc, gemini_api_key.strip())
                else:
                    st.session_state.advisor_response_payload = response_payload

    if st.session_state.advisor_error:
        st.error("Advisor pipeline failed.")
        st.code(st.session_state.advisor_error, language="text")

    if st.session_state.advisor_response_payload:
        st.caption(f"Last run: #{st.session_state.advisor_run_count}")
        render_response(st.session_state.advisor_response_payload)

    if st.session_state.last_request_payload:
        with st.expander("Last submitted payload"):
            st.json(st.session_state.last_request_payload)