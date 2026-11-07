"""Decision probability calibration with TypeSafe's Jev, when ``TYPESAFE_API_KEY``
is set.

Claude (the deep-think model) is the master decision maker: it reads the full
debate and commits to a rating. Jev is a different kind of model entirely — it
does not generate or argue, it only answers typed questions with calibrated
probabilities (see ``tradingagents.agents.post_screen`` for the sentiment-
screening use of the same ``system_one`` call). Here it is asked a single
``choice`` question over the same five-tier scale Claude just used, given the
same thesis, and returns a probability for every tier plus its own confidence.

This is independent corroboration, not a second vote: Jev never overrides or
replaces Claude's rating, and the calibration block is additive to the
decision, surfaced so a reader can see whether an independent, non-generative
model reads the thesis as decisively as Claude's own rating suggests, or finds
the call closer to a coin flip than the prose lets on.

Disabled by default (no ``TYPESAFE_API_KEY``), exactly like the sentiment
post-screen: callers get ``None`` back and skip the block entirely.
"""

from __future__ import annotations

import logging
import os

from tradingagents.agents.post_screen import TypeSafeError, system_one
from tradingagents.agents.schemas import PortfolioRating

logger = logging.getLogger(__name__)

# Keep the criteria in lockstep with PortfolioRating's own tiers rather than a
# hand-copied list: Jev's choice set and Claude's rating enum must always name
# the same five options, or probabilities and "stated rating" could silently
# drift into different vocabularies if the enum is ever extended.
_CRITERIA = {
    "Buy":         "The thesis supports strong conviction to enter or add to the position.",
    "Overweight":  "The thesis is favorable: gradually increasing exposure is supported.",
    "Hold":        "The thesis supports maintaining the current position, no action needed.",
    "Underweight": "The thesis supports reducing exposure or taking partial profits.",
    "Sell":        "The thesis supports exiting the position or avoiding entry.",
}

CALIBRATION_QUESTION = {
    "calibration": {
        "type": "choice",
        "instructions": (
            "Given `thesis`, which of the five position ratings does the evidence "
            "actually support? `stated_rating` is the rating a separate analysis "
            "already committed to; judge the thesis on its own merits rather than "
            "defaulting to it."
        ),
        "criteria": _CRITERIA,
    }
}


def jev_calibrate(thesis: str, stated_rating: str) -> dict | None:
    """An independent, calibrated probability distribution over the 5-tier
    rating scale, or ``None`` without a TypeSafe key or on any failure.

    ``thesis`` is the decision's own reasoning text (e.g. a Portfolio
    Manager's investment thesis); ``stated_rating`` is the rating it already
    committed to. Jev reads both as context but answers independently — a
    low probability on the stated rating, or a split distribution, is a
    useful signal precisely because Jev was never asked to agree.
    """
    if not os.environ.get("TYPESAFE_API_KEY"):
        return None
    try:
        answers = system_one(
            {"thesis": thesis, "stated_rating": stated_rating}, CALIBRATION_QUESTION
        )
        return answers["calibration"]
    except TypeSafeError as exc:
        logger.warning("Jev probability calibration unavailable: %s", exc)
        return None
    except (KeyError, TypeError) as exc:
        logger.warning("Jev probability calibration returned a malformed answer: %s", exc)
        return None


def render_calibration(answer: dict) -> str:
    """Render a Jev calibration answer as a markdown block for the decision text.

    Probabilities are listed in the rating scale's own order (most bullish to
    most bearish, matching ``PortfolioRating``/``RATINGS_5_TIER``), not sorted
    by size, so the distribution reads the same shape every run.
    """
    probabilities = answer.get("probabilities", {})
    chosen = answer.get("choice")
    confidence = answer.get("confidence")

    rows = ", ".join(
        f"{tier.value} {probabilities[tier.value]:.0%}"
        for tier in PortfolioRating
        if tier.value in probabilities
    )
    confidence_str = f"{confidence:.0%}" if isinstance(confidence, (int, float)) else str(confidence)

    return (
        "**Jev Probability Calibration** (independent, non-generative check): "
        f"{rows}. Jev's own pick: {chosen} (confidence: {confidence_str})."
    )
