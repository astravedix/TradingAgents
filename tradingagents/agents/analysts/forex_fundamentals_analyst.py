from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.turn import take_turn
from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.tools import (
    get_currency_scorecard,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
)
from tradingagents.dataflows.symbols import forex_pair_currencies

# The tools this analyst is offered; its tool node is built from the same tuple.
TOOLS = (
    get_currency_scorecard,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
)

SCORING_RUBRIC = """
Score each currency on a -5 (structurally weakest) to +5 (structurally
strongest) scale, across these four pillars. State the pillar score and a
one-line reason for each; then sum them for the currency's total.

1. Policy rate level + trajectory: a higher real policy rate and a hiking
   (or hawkish-paused) stance score higher; a lower rate and a cutting (or
   dovish-paused) stance score lower. Read the stance from the rate's recent
   direction in get_currency_scorecard AND from the tone of recent central
   bank statements/speeches in get_news and rate-decision odds in
   get_prediction_markets — a rate that is high but about to be cut is not
   the same as one expected to stay high.
2. Inflation (CPI YoY) trend: inflation closing in on target (~2% for most
   G10 central banks) from above, or stable at target, scores higher; inflation
   accelerating away from target, or undershooting it with disinflation risk,
   scores lower. Note the direction of travel, not just the level.
3. GDP growth momentum: accelerating, above-trend growth scores higher;
   decelerating or contracting growth scores lower.
4. Trade balance direction: an improving (more positive, or less negative)
   trade balance scores higher; a deteriorating one scores lower. A trade
   surplus is not automatically a higher score than a deficit if the deficit
   is narrowing and the surplus is shrinking — direction of travel matters
   over the level.

Total possible range per currency: -20 to +20. The pair's fundamental bias is
the base currency's total minus the quote currency's total: positive favors
the base currency, negative favors the quote currency.
"""


def create_forex_fundamentals_analyst(llm):
    def forex_fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)
        ticker = str(state["company_of_interest"])
        pair = forex_pair_currencies(ticker)
        if pair:
            base, quote = pair
            pair_instruction = (
                f"The pair under analysis is {base}/{quote} (base {base}, quote {quote}). "
                f"Call get_currency_scorecard once for '{base}' and once for '{quote}' "
                "before writing anything, so both sides of the comparison use the same "
                "as-of date and window."
            )
        else:
            # Shouldn't happen when this analyst is selected via the forex
            # asset-type path, but fails open rather than crashing the run.
            pair_instruction = (
                "Could not parse this ticker as a currency pair; call "
                "get_currency_scorecard for whichever ISO-4217 currencies are "
                "relevant, or report that the pair could not be identified."
            )

        system_message = (
            "You are a forex fundamentals researcher. Score the two currencies in the pair "
            "under analysis on central bank policy, inflation, growth, and trade balance, "
            "to give traders a structural (not short-term technical) read on the pair. "
            + pair_instruction
            + " Use get_news to check for recent central bank statements, speeches, or rate "
            "decisions that would color the policy-rate pillar's stance reading. Use "
            "get_prediction_markets (e.g. topic='<central bank> rate decision') for "
            "market-implied odds of the next move where relevant. Use get_macro_indicators "
            "with a raw FRED series ID for any data point get_currency_scorecard does not "
            "cover (it only covers USD, EUR, GBP, JPY, CAD, AUD, CHF, NZD)."
            + SCORING_RUBRIC
            + " Make sure to append a Markdown table at the end of the report with one row "
            "per currency per pillar (score + one-line reason), a totals row per currency, "
            "and a final line stating the net pair bias and which currency it favors."
            " Provide specific, actionable insights with supporting evidence to help traders "
            "make informed decisions. This is a structural fundamentals read, not a trade "
            "signal; say so explicitly and note that timing still depends on technicals and "
            "near-term catalysts this report does not cover."
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " Report what your tools support; another agent decides the trade."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges. {instrument_context}\n"
                    "{system_message}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in TOOLS]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        result, report = take_turn(prompt, llm, TOOLS, state["messages"])

        return {
            "messages": [result],
            "fundamentals_report": report,
        }

    return forex_fundamentals_analyst_node
