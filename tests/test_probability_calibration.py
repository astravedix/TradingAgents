"""Tests for the Jev decision-probability calibration layer.

Mirrors tests/test_post_screen.py's fixture style since both modules share
``system_one``; this file only tests what's specific to calibration:
question shape, the None-without-a-key path, failure handling, and rendering.
"""

import pytest

from tradingagents.agents import post_screen as typesafe
from tradingagents.agents import probability_calibration as calibration


class _Response:
    def __init__(self, status, payload=None):
        self.status_code = status
        self._payload = payload
        self.headers = {}

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON")
        return self._payload


class _Calls(list):
    response = None


@pytest.fixture
def post(monkeypatch):
    calls = _Calls()

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return calls.response

    monkeypatch.setattr(typesafe.requests, "post", fake_post)
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-test")
    monkeypatch.delenv("TYPESAFE_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    return calls


def _ok(choice="Buy", confidence=0.82, probabilities=None):
    probabilities = probabilities or {
        "Buy": 0.6, "Overweight": 0.2, "Hold": 0.1, "Underweight": 0.07, "Sell": 0.03,
    }
    return _Response(200, {
        "model": "jev-1.13.0",
        "answers": {
            "calibration": {
                "type": "choice",
                "choice": choice,
                "confidence": confidence,
                "probabilities": probabilities,
            }
        },
        "usage": {"input_tokens": 100, "output_tokens": 10},
    })


@pytest.mark.unit
def test_no_key_returns_none(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert calibration.jev_calibrate("thesis text", "Buy") is None


@pytest.mark.unit
def test_sends_the_thesis_and_stated_rating_as_state(post):
    post.response = _ok()

    result = calibration.jev_calibrate("Strong earnings beat, raising guidance.", "Overweight")

    url, kwargs = post[0]
    assert url == "https://api.typesafe.ai/v1/systemone"
    assert kwargs["json"]["state"] == {
        "thesis": "Strong earnings beat, raising guidance.",
        "stated_rating": "Overweight",
    }
    assert kwargs["json"]["questions"] == calibration.CALIBRATION_QUESTION
    assert result["choice"] == "Buy"


@pytest.mark.unit
def test_criteria_cover_exactly_the_five_tier_scale():
    from tradingagents.agents.schemas import PortfolioRating

    criteria = calibration.CALIBRATION_QUESTION["calibration"]["criteria"]
    assert set(criteria) == {r.value for r in PortfolioRating}


@pytest.mark.unit
def test_a_request_failure_returns_none_rather_than_raising(post):
    post.response = _Response(500)

    assert calibration.jev_calibrate("thesis", "Hold") is None


@pytest.mark.unit
def test_a_malformed_system_one_response_returns_none(post):
    # A response whose "calibration" answer type doesn't match the question's
    # declared type fails system_one's own check and raises TypeSafeError,
    # which jev_calibrate must swallow into None (never a tradeable surprise).
    post.response = _Response(200, {
        "model": "jev", "answers": {"calibration": {"type": "noul", "noul": 0.5}},
    })

    assert calibration.jev_calibrate("thesis", "Hold") is None


@pytest.mark.unit
def test_an_answer_missing_the_calibration_key_returns_none(post, monkeypatch):
    # Guards the KeyError path directly: system_one succeeds but the answers
    # dict it hands back doesn't contain what jev_calibrate asked for.
    monkeypatch.setattr(calibration, "system_one", lambda state, questions: {})

    assert calibration.jev_calibrate("thesis", "Hold") is None


@pytest.mark.unit
class TestRenderCalibration:
    def test_lists_probabilities_in_scale_order_not_sorted_by_size(self):
        md = calibration.render_calibration({
            "choice": "Overweight",
            "confidence": 0.75,
            "probabilities": {
                "Sell": 0.03, "Buy": 0.55, "Hold": 0.12,
                "Underweight": 0.05, "Overweight": 0.25,
            },
        })
        # Scale order is Buy, Overweight, Hold, Underweight, Sell regardless of
        # the dict's own (here deliberately scrambled) key order.
        assert md.index("Buy 55%") < md.index("Overweight 25%") < md.index("Hold 12%")
        assert md.index("Hold 12%") < md.index("Underweight 5%") < md.index("Sell 3%")
        assert "Jev's own pick: Overweight" in md
        assert "confidence: 75%" in md

    def test_non_numeric_confidence_is_rendered_as_is(self):
        md = calibration.render_calibration({
            "choice": "Hold", "confidence": "high", "probabilities": {"Hold": 1.0},
        })
        assert "confidence: high" in md
