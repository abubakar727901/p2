import os
import re
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import APIConnectionError, APIError, AuthenticationError, OpenAI, RateLimitError
from streamlit.errors import StreamlitSecretNotFoundError

from embeddings import chunk_text, create_embeddings
from pdf_reader import PDFProcessingError, extract_pdf_text
from rag import answer_question, retrieve_relevant_chunks


BASE_DIR = Path(__file__).resolve().parent
PDF_PATH = BASE_DIR / "restaurant_menu.pdf"
load_dotenv(BASE_DIR / ".env")


def _get_setting(name, default=""):
    value = os.getenv(name)
    if value is not None and value.strip():
        return value.strip()
    try:
        value = st.secrets.get(name, default)
    except StreamlitSecretNotFoundError:
        value = default
    return str(value).strip()


# This conservative mapping is the trusted ordering/pricing source and mirrors
# the supplied demonstration PDF exactly. The PDF remains the RAG knowledge source.
MENU = {
    "Biryani and Rice": {
        "Chicken Biryani": 350,
        "Beef Biryani": 450,
        "Chicken Pulao": 300,
        "Chicken Tikka Biryani": 450,
        "Special Zaiqa Palace Biryani": 550,
    },
    "Karahi and Handi": {
        "Chicken Karahi Half": 900,
        "Chicken Karahi Full": 1700,
        "Chicken White Karahi Half": 1100,
        "Chicken Handi": 1200,
        "Mutton Karahi Half": 1800,
    },
    "BBQ and Grill": {
        "Chicken Seekh Kebab, 2 pieces": 300,
        "Chicken Tikka Leg": 350,
        "Chicken Malai Boti": 550,
        "Beef Seekh Kebab, 2 pieces": 350,
        "Mixed BBQ Platter": 1400,
    },
    "Burgers and Fast Food": {
        "Zinger Burger": 550,
        "Chicken Cheese Burger": 650,
        "Beef Burger": 700,
        "Chicken Club Sandwich": 500,
        "French Fries": 250,
    },
    "Roti and Naan": {
        "Tandoori Roti": 40,
        "Plain Naan": 60,
        "Garlic Naan": 150,
        "Roghni Naan": 100,
    },
    "Drinks": {
        "Mineral Water": 100,
        "Soft Drink Regular": 120,
        "Fresh Lime": 180,
        "Mango Lassi": 250,
        "Fresh Orange Juice": 300,
    },
    "Desserts": {
        "Gulab Jamun, 2 pieces": 150,
        "Kheer": 200,
        "Gajar Ka Halwa": 250,
        "Kulfi": 150,
    },
}


def _init_state():
    defaults = {
        "messages": [],
        "cart": {},
        "confirmed_order": None,
        "order_name": "",
        "delivery_address": "",
        "service_type": "Pickup",
        "confirm_reviewed": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, (dict, list)) else value


@st.cache_data(show_spinner=False)
def _load_menu_index(pdf_path, modified_time, embedding_model):
    del modified_time
    text = extract_pdf_text(pdf_path)
    chunks = chunk_text(text)
    if not chunks:
        raise PDFProcessingError("No readable menu text was found in restaurant_menu.pdf.")
    api_key = _get_setting("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_openrouter_api_key_here":
        raise RuntimeError(
            "OpenRouter API key is missing. Set OPENROUTER_API_KEY in Streamlit Secrets "
            "or in your local .env file."
        )
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    vectors = create_embeddings(client, chunks, embedding_model)
    return chunks, vectors


def _menu_items():
    return {item: price for category in MENU.values() for item, price in category.items()}


def _cart_total(cart):
    return sum(_menu_items()[item] * quantity for item, quantity in cart.items() if item in _menu_items())


def _normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _find_order_items(text):
    normalized = _normalize(text)
    found = {}
    aliases = {
        "soft drink regular": ("soft drink", "soft drinks", "cold drink", "cold drinks"),
        "chicken seekh kebab, 2 pieces": ("chicken seekh kebab", "chicken seekh kabab"),
        "beef seekh kebab, 2 pieces": ("beef seekh kebab", "beef seekh kabab"),
        "gulab jamun, 2 pieces": ("gulab jamun",),
        "special zaiqa palace biryani": ("special biryani", "zaiqa palace biryani"),
    }
    items = sorted(_menu_items(), key=lambda name: len(_normalize(name)), reverse=True)
    for item in items:
        candidates = [item, *aliases.get(item.lower(), ())]
        for candidate in candidates:
            phrase = _normalize(candidate)
            if phrase and re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", normalized):
                quantity_match = re.search(
                    rf"(?<!\w)(\d+)\s+(?:x\s+)?{re.escape(phrase)}(?!\w)", normalized
                )
                quantity = int(quantity_match.group(1)) if quantity_match else 1
                if 1 <= quantity <= 99:
                    found[item] = max(found.get(item, 0), quantity)
                break
    return found


def _is_order_request(text):
    normalized = _normalize(text)
    return bool(
        _find_order_items(text)
        and re.search(
            r"\b(order|add|cart|chahiye|mangwa|mangwana|bhej|place|want|would like|get me|please)\b",
            normalized,
        )
    )


def _calculation_answer(question, retrieved):
    if not re.search(r"\b(total|kitna|kitne|how much|price|cost|amount)\b", _normalize(question)):
        return None
    quantities = _find_order_items(question)
    if not quantities:
        return None
    context = _normalize(" ".join(retrieved))
    context = re.sub(r"(?<=\d) (?=\d{3}\b)", "", context)
    items = _menu_items()
    for item in quantities:
        if _normalize(item) not in context or not re.search(
            rf"(?<!\d){items[item]}(?!\d)", context
        ):
            return None
    lines = [
        f"{quantity} × {item}: Rs. {items[item] * quantity:,}"
        for item, quantity in quantities.items()
    ]
    total = sum(items[item] * quantity for item, quantity in quantities.items())
    return f"Menu-based calculation:\n\n" + "\n\n".join(lines) + f"\n\n**Total: Rs. {total:,} PKR.**"


def _add_to_cart(item, quantity=1):
    if (
        st.session_state.confirmed_order
        or item not in _menu_items()
        or not isinstance(quantity, int)
        or quantity < 1
    ):
        return False
    st.session_state.cart[item] = min(99, st.session_state.cart.get(item, 0) + quantity)
    st.session_state.confirm_reviewed = False
    return True


def _set_cart_quantity(item, quantity):
    if st.session_state.confirmed_order or item not in _menu_items():
        return
    if quantity <= 0:
        st.session_state.cart.pop(item, None)
    else:
        st.session_state.cart[item] = min(99, quantity)
    st.session_state.confirm_reviewed = False


def _confirm_order():
    if st.session_state.confirmed_order or not st.session_state.cart:
        return False
    if st.session_state.service_type == "Delivery" and (
        not st.session_state.order_name.strip() or not st.session_state.delivery_address.strip()
    ):
        return False
    st.session_state.confirmed_order = {
        "reference": f"ZP-{uuid.uuid4().hex[:8].upper()}",
        "items": dict(st.session_state.cart),
        "total": _cart_total(st.session_state.cart),
        "service_type": st.session_state.service_type,
        "name": st.session_state.order_name.strip(),
        "address": st.session_state.delivery_address.strip(),
    }
    return True


def _ask_openrouter(question, chunks, vectors):
    api_key = _get_setting("OPENROUTER_API_KEY")
    if not api_key or api_key == "your_openrouter_api_key_here":
        raise RuntimeError(
            "OpenRouter API key is missing. Set OPENROUTER_API_KEY in Streamlit Secrets "
            "or in your local .env file."
        )
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    embedding_model = _get_setting(
        "OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small"
    )
    retrieved = retrieve_relevant_chunks(
        client, question, chunks, vectors, embedding_model=embedding_model
    )
    model = _get_setting("OPENROUTER_LLM_MODEL", "openai/gpt-4o-mini")
    response = answer_question(client, question, retrieved, model=model)
    return response, retrieved


def _render_cart():
    st.subheader("Your order")
    cart = st.session_state.cart
    order_confirmed = bool(st.session_state.confirmed_order)
    if not cart:
        st.caption("Your cart is empty. Add items from the menu below.")
    else:
        for index, (item, quantity) in enumerate(list(cart.items())):
            price = _menu_items()[item]
            st.markdown(f"**{item}**  ·  Rs. {price:,} each  ·  Subtotal Rs. {price * quantity:,}")
            left, middle, right = st.columns([1, 1, 1])
            with left:
                if st.button(
                    "−", key=f"dec_{index}", disabled=quantity <= 1 or order_confirmed
                ):
                    _set_cart_quantity(item, quantity - 1)
                    st.rerun()
            with middle:
                st.caption(f"Qty: {quantity}")
            with right:
                if st.button("Remove", key=f"remove_{index}", disabled=order_confirmed):
                    _set_cart_quantity(item, 0)
                    st.rerun()
            if st.button(
                "+ Increase", key=f"inc_{index}", disabled=quantity >= 99 or order_confirmed
            ):
                _set_cart_quantity(item, quantity + 1)
                st.rerun()
        st.markdown(f"### Total: Rs. {_cart_total(cart):,}")

    if order_confirmed:
        order = st.session_state.confirmed_order
        st.success(
            f"Order reference: **{order['reference']}**. This order is recorded in this session only; "
            "it has not been sent to the restaurant."
        )
        return

    st.session_state.service_type = st.radio(
        "Pickup or delivery?", ["Pickup", "Delivery"], key="service_choice"
    )
    if st.session_state.service_type == "Delivery":
        st.text_input("Customer name", key="order_name")
        st.text_area("Delivery address", key="delivery_address")
    if cart:
        with st.expander("Review order before confirmation", expanded=True):
            for item, quantity in cart.items():
                st.write(f"{quantity} × {item} — Rs. {_menu_items()[item] * quantity:,}")
            st.write(f"**Grand total: Rs. {_cart_total(cart):,}**")
            st.caption(f"Service: {st.session_state.service_type}")
        st.checkbox("I have reviewed this order and want to confirm it.", key="confirm_reviewed")
    disabled = (
        not cart
        or not st.session_state.confirm_reviewed
        or (
            st.session_state.service_type == "Delivery"
            and (not st.session_state.order_name.strip() or not st.session_state.delivery_address.strip())
        )
    )
    if st.button("Confirm order", type="primary", disabled=disabled, key="confirm_order"):
        if _confirm_order():
            st.rerun()
        else:
            st.error("Please add at least one menu item and complete the required delivery details.")


def _render_sidebar():
    with st.sidebar:
        st.markdown("## Zaiqa Palace")
        st.caption("Pakistani cuisine · Prices in PKR")
        st.caption("AI is powered by OpenRouter. Configure OPENROUTER_API_KEY in Streamlit Secrets or .env.")
        st.info(
            "Demonstration menu only; these sample prices are not represented as prices of a real restaurant."
        )
        st.markdown("### Menu categories")
        for category, items in MENU.items():
            with st.expander(category):
                for item, price in items.items():
                    st.write(f"{item} — Rs. {price:,}")
        st.markdown("### Add an item")
        all_items = _menu_items()
        selected_item = st.selectbox(
            "Menu item", list(all_items), key="selected_menu_item"
        )
        selected_quantity = st.number_input(
            "Quantity", min_value=1, max_value=99, value=1, step=1, key="selected_item_quantity"
        )
        if st.button("Add selected item", use_container_width=True):
            if _add_to_cart(selected_item, int(selected_quantity)):
                st.rerun()
            else:
                st.error("A confirmed order cannot be changed. Start a new conversation to make another order.")
        st.caption(
            "Ordering uses a conservative built-in menu/pricing map that must match the PDF. "
            "Orders are session-only and are not automatically sent to the restaurant."
        )
        _render_cart()
        if st.button("Clear chat and start new conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.cart = {}
            st.session_state.confirmed_order = None
            st.session_state.confirm_reviewed = False
            st.session_state.order_name = ""
            st.session_state.delivery_address = ""
            st.rerun()


def main():
    st.set_page_config(page_title="Zaiqa Palace | Restaurant Chat", page_icon="🍽️", layout="wide")
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(135deg, #fffaf0 0%, #f5f7ef 55%, #fff 100%); }
        [data-testid="stSidebar"] { background: #173d2b; }
        [data-testid="stSidebar"] * { color: #fffaf0; }
        [data-testid="stSidebar"] .stButton button { color: #173d2b; }
        .hero { padding: 1.5rem 1.8rem; border-radius: 18px; background: #173d2b; color: #fffaf0; }
        .hero h1 { color: #f4c46b; margin-bottom: .2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    _init_state()
    _render_sidebar()
    st.markdown(
        '<div class="hero"><h1>Zaiqa Palace</h1>'
        "<p>Welcome! Ask about our Pakistani menu in English or Roman Urdu, or tell me what you would like to order.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("context"):
                with st.expander("Retrieved menu context"):
                    for chunk in message["context"]:
                        st.markdown(f"> {chunk}")

    prompt = st.chat_input("Ask in English or Roman Urdu…")
    if prompt is None:
        return
    prompt = prompt.strip()
    if not prompt:
        st.warning("Please enter a question or order request.")
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            if _is_order_request(prompt):
                requested = _find_order_items(prompt)
                if st.session_state.confirmed_order:
                    answer = (
                        "This order is already finalized for this session and cannot be changed. "
                        "Start a new conversation to prepare another draft order."
                    )
                else:
                    for item, quantity in requested.items():
                        _add_to_cart(item, quantity)
                    answer = (
                        "I added the requested menu item(s) to your draft cart. Please review the quantities "
                        "and total in the sidebar. This is not a confirmed order."
                    )
                context = []
            else:
                with st.spinner("Checking the menu and preparing a reply…"):
                    if not PDF_PATH.is_file():
                        raise PDFProcessingError(
                            "The permanent menu file restaurant_menu.pdf is missing from the project folder."
                        )
                    modified_time = PDF_PATH.stat().st_mtime_ns
                    embedding_model = _get_setting(
                        "OPENROUTER_EMBEDDING_MODEL",
                        "openai/text-embedding-3-small",
                    )
                    chunks, vectors = _load_menu_index(
                        str(PDF_PATH), modified_time, embedding_model
                    )
                    answer, context = _ask_openrouter(prompt, chunks, vectors)
                    answer = _calculation_answer(prompt, context) or answer
            st.markdown(answer)
            if context:
                with st.expander("Retrieved menu context"):
                    for chunk in context:
                        st.markdown(f"> {chunk}")
        except PDFProcessingError as exc:
            answer, context = f"Menu PDF problem: {exc}", []
            st.error(answer)
        except RuntimeError as exc:
            answer, context = str(exc), []
            st.error(answer)
        except AuthenticationError:
            answer, context = (
                "OpenRouter rejected the API credentials. Check your OPENROUTER_API_KEY in Streamlit Secrets or .env.",
                [],
            )
            st.error(answer)
        except RateLimitError:
            answer, context = "The AI service is rate-limited right now. Please wait a moment and try again.", []
            st.error(answer)
        except APIConnectionError:
            answer, context = "Could not connect to the AI service. Check your connection and try again.", []
            st.error(answer)
        except APIError:
            answer, context = "The AI service could not complete this request. Please try again shortly.", []
            st.error(answer)
        except ValueError as exc:
            answer, context = f"Menu or question could not be processed: {exc}", []
            st.error(answer)
        except OSError:
            answer, context = (
                "A local menu file could not be accessed. Check restaurant_menu.pdf and try again.",
                [],
            )
            st.error(answer)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "context": context}
    )
    st.rerun()


if __name__ == "__main__":
    main()
