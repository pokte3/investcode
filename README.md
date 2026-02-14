 (cd "$(git rev-parse --show-toplevel)" && git apply --3way <<'EOF' 
diff --git a/README.md b/README.md
index 6fdff47ba4ea85171ebf4c0fb4af8d06f9b61fc3..6e8b159353690d157663abebee204dbf88f8f34d 100644
--- a/README.md
+++ b/README.md
@@ -1 +1,41 @@
-# investcode
\ No newline at end of file
+# investcode
+
+추세추종(TSMOM/이동평균·돌파) + 모멘텀 기반 진입 신호를 **지도학습 형태로 보정**하는 예제입니다.
+
+핵심 아이디어:
+- 기본 진입은 추세추종으로 잡습니다.
+- 과거 유사 패턴에서 다음 봉 수익률이 나빴던 경우를 학습합니다.
+- "진입각"이어도 손실 확률이 높은 패턴이면 스킵합니다.
+
+## 포함된 구성
+- `Candle`: OHLCV 데이터 구조
+- `build_dataset`: 다음 봉 수익률(라벨) + 추세/모멘텀/돌파 특징 생성
+- `SetupOutcomeModel`: 패턴 버킷별 평균 다음 봉 수익률을 학습해서 진입 여부 결정
+
+## 빠른 사용 예시
+
+```python
+from main import Candle, SetupOutcomeModel, build_dataset
+
+candles = [
+    Candle(open=100, high=101, low=99, close=100.5),
+    # ... 과거 캔들 추가
+]
+
+model = SetupOutcomeModel(min_samples=4)
+model.fit(candles, short_ma=5, long_ma=20)
+
+current_feature = {
+    "trend_strength": 0.012,
+    "momentum_lookback": 0.035,
+    "breakout_distance": 0.008,
+}
+
+decision = model.decide_entry(current_feature, min_expected_return=0.001)
+print(decision.enter, decision.predicted_return, decision.reason)
+```
+
+## 커맨드
+- Setup: `python -m pip install -r requirements.txt`
+- Lint: `ruff check .`
+- Test: `pytest -q`
 
EOF
)
