import os
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from tool_logic import search_listings_impl

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


@tool
def search_listings(
    block: Optional[str] = None,
    plot: Optional[str] = None,
    land_use: Optional[str] = None,
    property_type: Optional[str] = None,
    min_price_crore: Optional[float] = None,
    max_price_crore: Optional[float] = None,
    min_marla: Optional[float] = None,
    max_marla: Optional[float] = None,
    sort_by: str = "price_asc",
    limit: int = 10,
) -> str:
    """Search LIVE DHA Lahore Phase 8 for-sale listings. Leave a filter empty to not filter on it.
    block: e.g. "T", "V", "CCA-1". plot: plot number. land_use: "Residential" or "Commercial".
    property_type: "Plots" or "Houses". Prices are in crore (1 crore = 10,000,000 PKR).
    Size is in marla (1 kanal = 20 marla, so 1 kanal -> min_marla=20, max_marla=20).
    sort_by: price_asc, price_desc, size_asc, size_desc, price_per_marla_asc.
    For "cheapest" use price_asc with limit=1; for "most expensive" use price_desc."""
    return search_listings_impl(block, plot, land_use, property_type, min_price_crore,
                                max_price_crore, min_marla, max_marla, sort_by, limit)


SYSTEM = (
    "You are a property assistant for DHA Lahore Phase 8 plots and houses for sale. "
    "ALWAYS call search_listings to get live data before answering about listings; never invent listings. "
    "Give block, plot number, size and price in crore. If a result has a NOTE, tell the user to verify that with the seller. "
    "If the tool says there are more results than shown, tell the user the total count. "
    "If nothing matches, say so and suggest loosening a filter. "
    "Reply in the same language the user writes in (English or Roman Urdu)."
)

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model=MODEL, temperature=0).bind_tools([search_listings])
    return _llm


def ask(question: str) -> dict:
    messages = [SystemMessage(SYSTEM), HumanMessage(question)]
    calls = []
    ai = None
    for _ in range(4):  # max 4 tool rounds
        ai = get_llm().invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            break
        for call in ai.tool_calls:
            calls.append(call["args"])
            result = search_listings.invoke(call["args"])
            messages.append(ToolMessage(result, tool_call_id=call["id"]))
    return {"answer": ai.content, "searches": calls}
