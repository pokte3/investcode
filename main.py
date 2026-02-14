 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/main.py b/main.py
index 4693ad3cf8b0903b98497fb89b8b524fbf1b93f4..9116cc7947c549c18c87b871d7fb790bbc739990 100644
--- a/main.py
+++ b/main.py
@@ -1,2 +1,170 @@
-def add(a, b):
+from __future__ import annotations
+
+from collections import defaultdict
+from dataclasses import dataclass
+from statistics import mean
+from typing import Iterable
+
+
+@dataclass(frozen=True)
+class Candle:
+    """Simple OHLCV candle."""
+
+    open: float
+    high: float
+    low: float
+    close: float
+    volume: float = 0.0
+
+
+@dataclass(frozen=True)
+class TradeDecision:
+    """Model output for a single setup."""
+
+    enter: bool
+    predicted_return: float
+    reason: str
+
+
+class SetupOutcomeModel:
+    """Supervised learner that predicts next-candle return from setup features.
+
+    The model is intentionally light-weight:
+    - builds trend-following + momentum features
+    - buckets features into coarse "patterns"
+    - learns average next-candle return per pattern
+
+    This lets you answer:
+    "Even if this looks like trend-follow entry, was this pattern often loss-making?"
+    """
+
+    def __init__(self, min_samples: int = 4):
+        self.min_samples = min_samples
+        self._bucket_returns: dict[tuple[int, int, int], list[float]] = defaultdict(list)
+        self._global_mean: float = 0.0
+        self._fitted = False
+
+    def fit(self, candles: Iterable[Candle], short_ma: int = 5, long_ma: int = 20) -> None:
+        candles = list(candles)
+        if len(candles) <= long_ma:
+            raise ValueError("Not enough candles for training.")
+
+        features, labels = build_dataset(candles, short_ma=short_ma, long_ma=long_ma)
+        if not labels:
+            raise ValueError("Could not build training labels.")
+
+        self._bucket_returns.clear()
+        self._global_mean = mean(labels)
+
+        for feature, label in zip(features, labels):
+            bucket = self._to_bucket(feature)
+            self._bucket_returns[bucket].append(label)
+
+        self._fitted = True
+
+    def predict_next_return(self, feature: dict[str, float]) -> float:
+        self._require_fitted()
+        bucket = self._to_bucket(feature)
+        values = self._bucket_returns.get(bucket)
+
+        if not values:
+            return self._global_mean
+        if len(values) < self.min_samples:
+            # low-confidence pattern: blend with global mean
+            return 0.5 * mean(values) + 0.5 * self._global_mean
+        return mean(values)
+
+    def decide_entry(self, feature: dict[str, float], min_expected_return: float = 0.0) -> TradeDecision:
+        prediction = self.predict_next_return(feature)
+        if prediction > min_expected_return:
+            return TradeDecision(
+                enter=True,
+                predicted_return=prediction,
+                reason="과거 유사 패턴의 다음 봉 기대수익률이 기준보다 높습니다.",
+            )
+        return TradeDecision(
+            enter=False,
+            predicted_return=prediction,
+            reason="유사 패턴에서 손실/저수익 확률이 높아 진입을 건너뜁니다.",
+        )
+
+    def _to_bucket(self, feature: dict[str, float]) -> tuple[int, int, int]:
+        trend_score = _bin(feature["trend_strength"], [-0.02, 0.0, 0.02])
+        momentum_score = _bin(feature["momentum_lookback"], [-0.03, 0.0, 0.03])
+        breakout_score = _bin(feature["breakout_distance"], [-0.01, 0.0, 0.01])
+        return trend_score, momentum_score, breakout_score
+
+    def _require_fitted(self) -> None:
+        if not self._fitted:
+            raise RuntimeError("Model is not fitted.")
+
+
+def add(a: float, b: float) -> float:
     return a + b
+
+
+def moving_average(values: list[float], window: int) -> list[float]:
+    if window <= 0:
+        raise ValueError("window must be positive")
+    if len(values) < window:
+        return []
+
+    out: list[float] = []
+    running = sum(values[:window])
+    out.append(running / window)
+
+    for idx in range(window, len(values)):
+        running += values[idx] - values[idx - window]
+        out.append(running / window)
+    return out
+
+
+def build_dataset(
+    candles: list[Candle], short_ma: int = 5, long_ma: int = 20
+) -> tuple[list[dict[str, float]], list[float]]:
+    closes = [c.close for c in candles]
+    highs = [c.high for c in candles]
+
+    short_series = moving_average(closes, short_ma)
+    long_series = moving_average(closes, long_ma)
+    if not short_series or not long_series:
+        return [], []
+
+    features: list[dict[str, float]] = []
+    labels: list[float] = []
+
+    start = long_ma - 1
+    for idx in range(start, len(candles) - 1):
+        short_idx = idx - (short_ma - 1)
+        long_idx = idx - (long_ma - 1)
+
+        short_now = short_series[short_idx]
+        long_now = long_series[long_idx]
+        trend_strength = (short_now - long_now) / long_now if long_now else 0.0
+
+        lookback = 10
+        momentum_ref_idx = max(0, idx - lookback)
+        momentum_lookback = (closes[idx] - closes[momentum_ref_idx]) / closes[momentum_ref_idx]
+
+        recent_high = max(highs[max(0, idx - 20) : idx + 1])
+        breakout_distance = (closes[idx] - recent_high) / recent_high if recent_high else 0.0
+
+        next_return = (closes[idx + 1] - closes[idx]) / closes[idx]
+
+        features.append(
+            {
+                "trend_strength": trend_strength,
+                "momentum_lookback": momentum_lookback,
+                "breakout_distance": breakout_distance,
+            }
+        )
+        labels.append(next_return)
+
+    return features, labels
+
+
+def _bin(value: float, cut_points: list[float]) -> int:
+    for index, cut in enumerate(cut_points):
+        if value < cut:
+            return index
+    return len(cut_points)
 
EOF
)
