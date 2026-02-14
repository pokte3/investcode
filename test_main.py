 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/test_main.py b/test_main.py
index dd3eebd71574bbe04e0960b192b61188db90b572..8294d36aba854c333f117bd4d948e60d46925442 100644
--- a/test_main.py
+++ b/test_main.py
@@ -1,4 +1,45 @@
-from main import add
+from main import Candle, SetupOutcomeModel, add, build_dataset
+
 
 def test_add():
     assert add(1, 2) == 3
+
+
+def _make_uptrend_with_pullbacks(length: int = 80) -> list[Candle]:
+    candles = []
+    price = 100.0
+    for i in range(length):
+        drift = 0.8 if i % 7 else -1.2  # occasional pullback candles
+        close = max(1.0, price + drift)
+        high = max(price, close) + 0.5
+        low = min(price, close) - 0.5
+        candles.append(Candle(open=price, high=high, low=low, close=close, volume=1000))
+        price = close
+    return candles
+
+
+def test_build_dataset_returns_features_and_labels():
+    candles = _make_uptrend_with_pullbacks()
+    features, labels = build_dataset(candles, short_ma=5, long_ma=20)
+
+    assert features
+    assert labels
+    assert len(features) == len(labels)
+    assert {"trend_strength", "momentum_lookback", "breakout_distance"}.issubset(features[0])
+
+
+def test_model_can_block_bad_patterns():
+    candles = _make_uptrend_with_pullbacks()
+    model = SetupOutcomeModel(min_samples=2)
+    model.fit(candles)
+
+    # Artificially bad pattern: strong trend but late breakout overextension
+    bad_like_feature = {
+        "trend_strength": 0.01,
+        "momentum_lookback": 0.10,
+        "breakout_distance": 0.03,
+    }
+
+    decision = model.decide_entry(bad_like_feature, min_expected_return=0.005)
+    assert decision.enter is False
+    assert decision.predicted_return <= 0.005
 
EOF
)
