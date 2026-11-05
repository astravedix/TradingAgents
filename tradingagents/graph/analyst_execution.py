from collections.abc import Iterable
from dataclasses import dataclass

from tradingagents.agents.analysts import (
    forex_fundamentals_analyst,
    fundamentals_analyst,
    market_analyst,
    news_analyst,
)


@dataclass(frozen=True)
class AnalystNodeSpec:
    key: str
    agent_node: str
    report_key: str
    tools: tuple = ()


@dataclass(frozen=True)
class AnalystExecutionPlan:
    specs: list[AnalystNodeSpec]


ANALYST_NODE_SPECS: dict[str, AnalystNodeSpec] = {
    "market": AnalystNodeSpec(
        key="market",
        agent_node="Market Analyst",
        report_key="market_report",
        tools=market_analyst.TOOLS,
    ),
    "social": AnalystNodeSpec(
        # Saved configs select this analyst as "social". It fetches its
        # sources before calling the model, so it has no tools.
        key="social",
        agent_node="Sentiment Analyst",
        report_key="sentiment_report",
    ),
    "news": AnalystNodeSpec(
        key="news",
        agent_node="News Analyst",
        report_key="news_report",
        tools=news_analyst.TOOLS,
    ),
    "fundamentals": AnalystNodeSpec(
        key="fundamentals",
        agent_node="Fundamentals Analyst",
        report_key="fundamentals_report",
        tools=fundamentals_analyst.TOOLS,
    ),
}

# The "fundamentals" key means something different for a currency pair than
# for a company: there is no balance sheet to read, so asset_type="forex"
# swaps in the Forex Fundamentals Analyst's own tools (central bank policy,
# inflation, growth, trade balance) under the same key, agent node name and
# report slot. Selection (`--analysts fundamentals`) and the state's
# fundamentals_report field stay the same either way; only what runs under
# them changes.
_FOREX_FUNDAMENTALS_SPEC = AnalystNodeSpec(
    key="fundamentals",
    agent_node="Fundamentals Analyst",
    report_key="fundamentals_report",
    tools=forex_fundamentals_analyst.TOOLS,
)


def build_analyst_execution_plan(
    selected_analysts: Iterable[str],
    asset_type: str = "stock",
) -> AnalystExecutionPlan:
    specs: list[AnalystNodeSpec] = []
    for analyst_key in selected_analysts:
        if analyst_key == "fundamentals" and asset_type == "forex":
            specs.append(_FOREX_FUNDAMENTALS_SPEC)
            continue
        spec = ANALYST_NODE_SPECS.get(analyst_key)
        if spec is None:
            raise ValueError(f"unknown analyst key: {analyst_key}")
        specs.append(spec)

    if not specs:
        raise ValueError("at least one analyst must be selected")

    return AnalystExecutionPlan(specs=specs)


