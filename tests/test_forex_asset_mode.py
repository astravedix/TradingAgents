import unittest

from cli.models import AnalystType, AssetType
from cli.prompts import detect_asset_type, filter_analysts_for_asset_type
from tradingagents.agents.analysts import forex_fundamentals_analyst, fundamentals_analyst
from tradingagents.dataflows.symbols import forex_pair_currencies
from tradingagents.dataflows.vendors.fred import CURRENCY_PILLARS, _resolve_series_id
from tradingagents.graph.analyst_execution import build_analyst_execution_plan
from tradingagents.graph.propagation import Propagator


class ForexAssetModeTests(unittest.TestCase):
    def test_detects_forex_pair_symbols(self):
        self.assertEqual(detect_asset_type("EURUSD"), AssetType.FOREX)
        self.assertEqual(detect_asset_type("eurusd"), AssetType.FOREX)
        self.assertEqual(detect_asset_type("GBPJPY"), AssetType.FOREX)
        self.assertEqual(detect_asset_type("USDCAD+"), AssetType.FOREX)

    def test_does_not_misclassify_crypto_or_metals_as_forex(self):
        # BTC-USD/XAUUSD resolve through other rules first and must stay as-is.
        self.assertEqual(detect_asset_type("BTC-USD"), AssetType.CRYPTO)
        self.assertEqual(detect_asset_type("XAUUSD"), AssetType.STOCK)

    def test_defaults_non_forex_symbols_to_stock(self):
        self.assertEqual(detect_asset_type("AAPL"), AssetType.STOCK)

    def test_keeps_all_analysts_selectable_for_forex(self):
        analysts = [
            AnalystType.MARKET,
            AnalystType.SOCIAL,
            AnalystType.NEWS,
            AnalystType.FUNDAMENTALS,
        ]
        self.assertEqual(
            filter_analysts_for_asset_type(analysts, AssetType.FOREX),
            analysts,
        )

    def test_propagator_includes_forex_asset_type_in_initial_state(self):
        state = Propagator().create_initial_state(
            "EURUSD=X", "2026-04-18", asset_type=AssetType.FOREX.value
        )
        self.assertEqual(state["asset_type"], AssetType.FOREX.value)

    def test_forex_pair_currencies_parses_known_forms(self):
        self.assertEqual(forex_pair_currencies("EURUSD=X"), ("EUR", "USD"))
        self.assertEqual(forex_pair_currencies("gbpjpy"), ("GBP", "JPY"))
        self.assertEqual(forex_pair_currencies("USDCAD+"), ("USD", "CAD"))
        self.assertIsNone(forex_pair_currencies("AAPL"))
        self.assertIsNone(forex_pair_currencies("BTC-USD"))

    def test_execution_plan_swaps_fundamentals_tools_for_forex(self):
        plan = build_analyst_execution_plan(["fundamentals"], asset_type="forex")
        spec = plan.specs[0]
        self.assertEqual(spec.agent_node, "Fundamentals Analyst")
        self.assertEqual(spec.report_key, "fundamentals_report")
        self.assertEqual(spec.tools, forex_fundamentals_analyst.TOOLS)
        self.assertNotEqual(spec.tools, fundamentals_analyst.TOOLS)

    def test_execution_plan_keeps_equity_fundamentals_for_stock(self):
        plan = build_analyst_execution_plan(["fundamentals"], asset_type="stock")
        self.assertEqual(plan.specs[0].tools, fundamentals_analyst.TOOLS)

    def test_every_currency_pillar_resolves_to_a_fred_series_id(self):
        # Every alias must resolve (no typos/raw-ID drift) and every currency
        # must cover the same four pillars.
        expected_pillars = {"policy_rate", "cpi", "gdp_growth", "trade_balance"}
        for currency, pillars in CURRENCY_PILLARS.items():
            with self.subTest(currency=currency):
                self.assertEqual(set(pillars), expected_pillars)
                for alias in pillars.values():
                    series_id = _resolve_series_id(alias)
                    self.assertTrue(series_id.isalnum(), series_id)


if __name__ == "__main__":
    unittest.main()
