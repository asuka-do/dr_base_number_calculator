# DR ポートフォリオ最適化シミュレータ

需給調整市場向けに、基地局蓄電池をアグリゲートしてDR（デマンドレスポンス）目標を達成するための必要局数 N\* を確率的に求めるシミュレータです。

## 概要

多数の基地局に分散配置された蓄電池（kW・kWhがばらつく）を1つのポートフォリオとして扱い、指定されたDR目標を高信頼度（99.95%）で達成するために必要な最小局数を Sample Average Approximation (SAA) + 二分探索で算出します。

- **入力**: kW/kWh分布、故障率、DR目標、継続時間
- **出力**: 必要局数 N\* とその不確実性、稼働/退場/未使用の内訳（4種類のグラフ）

## 必要環境

- Python 3.8+
- 依存パッケージ

```bash
pip install numpy scipy matplotlib japanize_matplotlib
```

## 実行方法

```bash
python dr_simulator_v2.py
```

出力先はデフォルトで `/mnt/user-data/outputs`。Windows/Macでは CONFIG の `OUT` を変更してください（例: `OUT = "./output"`）。

## 出力ファイル

| ファイル | 内容 |
|---|---|
| `fig1v2_fleet_distribution.png` | kW/kWh/持続時間の分布 + kW-kWh相関散布図 |
| `fig2v2_composition_7panels.png` | 期待値ベースの kW・kWh別出力構成（7つのDR目標） |
| `fig3v2_composition_stochastic_7panels.png` | 確率的版（300試行の中央値・25-75%ile・10-90%ile） |
| `fig4v2_deviation_combo.png` | 稼働/退場/未使用の3分類（kWh別） |

## 数理モデル

### 蓄電池モデル

各蓄電池 $i$ は3つの属性を持つ：

- $\mathrm{kW}_i$: 定格出力（離散正規分布）
- $\mathrm{kWh}_i$: 定格容量（離散正規分布・不等間隔ラインナップ）
- $A_i \sim \mathrm{Bernoulli}(1-p_\mathrm{fail})$: 稼働ステータス（故障率 6%）

時刻 $t$ における出力：

$$
\text{出力}_i(t) = \mathrm{kW}_i \times \mathbb{I}(\mathrm{dur}_i > t + \text{⑦}) \times A_i
$$

ここで $\mathrm{dur}_i = \mathrm{kWh}_i / \mathrm{kW}_i$、$\text{⑦}=3\mathrm{h}$（最低持続時間）。

### N\* の算出

DR目標 $q$、DR継続時間 $T$、信頼度 $1-\alpha$（デフォルト 99.95%）に対する最小局数：

$$
N^*(q) = \min \left\lbrace N \,\middle|\, \Pr\left[\sum_{i=1}^{N} \text{出力}_i(T) \geq q_\text{lo}\right] \geq 1 - \alpha \right\rbrace
$$

ここで達成下限 $q_\text{lo}$ は成功幅 $r$（デフォルト 0.10）を用いて：

$$
q_\text{lo} = q \times (1 - r)
$$

例：$q = 1000\text{kW}$, $r = 0.10$ の場合 $q_\text{lo} = 900\text{kW}$。

SAA（Sample Average Approximation）+ 二分探索で計算します。

### kW-kWh 正相関

ガウスコピュラ（$\rho = 0.75$）で kW と kWh の間に正相関を持たせています（大kW → 大kWh の傾向）。

## パラメータ設定

`dr_simulator_v2.py` の CONFIG セクション（30-140行目）ですべての定数を集中管理しています。

### 主要パラメータ

#### 蓄電池分布

```python
# kW分布（離散正規 [1,2,...,10] kW）
KW_MIN, KW_MAX, KW_STEP = 1.0, 10.0, 1.0
KW_MEDIAN, KW_STD = 5.0, 2.0

# kWh分布（不等間隔ラインナップ）
KWH_LINEUP = [15., 30., 37.5, 40., 60., 75.]
KWH_MEDIAN = 30.0    # LINEUP内の値
KWH_SPREAD = 1.0     # index空間の広がり
```

#### 物理パラメータ

```python
SUCCESS_RANGE = 0.10   # DR達成幅 ±10%
MIN_DUR       = 3.0    # 最低持続時間 ⑦ [h]
T_BASE        = 3.0    # SAA基準のDR継続時間 [h]
T_FIG         = 6.0    # グラフ用のDR継続時間 [h]
KW_CORR       = 0.75   # kW-kWh正相関（コピュラρ）
FAILURE_RATE  = 0.06   # 故障率
```

#### DR目標

```python
Q_TARGETS   = [100, 250, 500, 750, 1000, 1250, 1500]
Q_NSTAR_REF = 1000    # N* 精密算出の基準DR目標 [kW]
```

#### 精度設定

```python
NT_NSTAR_FAST     = 1500  # bootstrap内の各N*探索の試行数
NSTAR_BOOTSTRAP_K = 50    # N*σ評価のbootstrap反復数
```

高速化したい場合は `NT_NSTAR_FAST = 300, NSTAR_BOOTSTRAP_K = 20` に変更（速度優先）。

## 実行時出力

```
プール生成中...
  20,000局生成完了（dur>3.0h でフィルタ済み）
  ── 設定値 vs 実現値 ──
    kW : 設定 中央値=5.00, σ=2.00
         実現 中央値=5.00, 平均=5.02, σ=1.94
    kWh: 設定 中央値=30.00, spread(idx)=1.00
         実現 中央値=30.00, 平均=29.51, σ=8.70
    dur: 実現 中央値=6.00h, 平均=6.47h, min=3.33h

⑤ N* 探索中 (1000kW, T=3.0h)...
  N*(1000kW) = 709局  達成=99.85%  失敗=1/1000

  全DR目標のN*計算（50回bootstrap × 各1500試行）...
       N* 平均 ±    σ    最小-  最大
    N*(  100kW) =  105.2 ±  3.8局   (  97- 117)   採用=105局
    N*(  250kW) =  216.1 ±  6.9局   ( 204- 230)   採用=216局
    N*(  500kW) =  385.7 ±  7.1局   ( 372- 401)   採用=386局
    N*(  750kW) =  553.0 ±  8.3局   ( 534- 574)   採用=553局
    N*( 1000kW) =  718.2 ± 11.0局   ( 699- 746)   採用=718局
    N*( 1250kW) =  878.5 ±  9.5局   ( 860- 899)   採用=878局
    N*( 1500kW) = 1038.2 ± 13.0局   (1016-1062)   採用=1038局
```

計算時間は約50秒（精度優先設定）です。

## パラメータ変更例

### 例1: 蓄電池を大容量化

より大容量な蓄電池を使う場合：

```python
KWH_MEDIAN = 40.0   # 中央値を40kWhに
```

→ 持続時間が伸び、必要局数が大幅減少（1000kW: ~700局 → ~230局）

### 例2: 故障率を変える

信頼性の高い機器なら故障率を下げる：

```python
FAILURE_RATE = 0.02   # 6% → 2%
```

### 例3: kW刻みを細かく

```python
KW_STEP = 0.5   # 0.5kW刻み [1.0, 1.5, 2.0, ..., 10.0]
```

## Fig4 の3分類定義

すべて逸脱ラインの次の離散時間点 $t_\text{dev}$ で評価。

**稼働数**: 使用され、持続時間⑦以上残っている局

$$
n_\text{active} = N_\text{use} \times (1 - p_\text{fail}) \times \Pr(\mathrm{dur} > t_\text{dev} + \text{⑦})
$$

**退場数**: 使用され、持続時間⑦に到達（または故障）した局

$$
n_\text{retired} = N_\text{use} \times \left[ p_\text{fail} + (1 - p_\text{fail}) \times \Pr(\mathrm{dur} \leq t_\text{dev} + \text{⑦}) \right]
$$

**未使用数**: DR目標が小さく、余剰として使用されていない局

$$
n_\text{unused} = N^* - N_\text{use}
$$

ただし $N_\text{use} = \min(N^*, N^*(q))$。合計は常に $N^*$ となる（$n_\text{active} + n_\text{retired} + n_\text{unused} = N^*$）。

## Git 運用

### 初期化

```bash
git init
git add .
git commit -m "Initial commit: DR portfolio optimization simulator v2"
```

### ブランチ運用（推奨）

```bash
# パラメータ実験用ブランチ
git checkout -b experiment/kwh-median-40

# パラメータ変更後
git add dr_simulator_v2.py
git commit -m "Change KWH_MEDIAN to 40 for large-capacity battery scenario"
```

### 出力ファイルの管理

生成される PNG ファイルは Git 管理対象外（`.gitignore` で除外）。結果を残したい場合は個別に `git add -f` で強制追加：

```bash
git add -f fig1v2_fleet_distribution.png   # 特別な結果だけ保存
```

## ディレクトリ構成

```
.
├── dr_simulator_v2.py    # メインスクリプト
├── README.md             # このファイル
├── .gitignore            # Git除外設定
└── output/               # 生成グラフ（gitignore対象）
    ├── fig1v2_fleet_distribution.png
    ├── fig2v2_composition_7panels.png
    ├── fig3v2_composition_stochastic_7panels.png
    └── fig4v2_deviation_combo.png
```

## トラブルシューティング

### `ModuleNotFoundError: No module named 'japanize_matplotlib'`

```bash
pip install japanize_matplotlib
```

### `AssertionError: KWH_MEDIAN=XX は KWH_LINEUP に含まれる値でなければなりません`

`KWH_MEDIAN` は `KWH_LINEUP` に必ず含まれる値にする。LINEUPを変えた場合は両方セットで更新。

### 出力ディレクトリが存在しない

CONFIG の `OUT` を実在パスに変更するか、`os.makedirs(OUT, exist_ok=True)` が自動作成するのでそのまま動作。

### 計算が遅い

`NT_NSTAR_FAST = 300`, `NSTAR_BOOTSTRAP_K = 20` に下げる（結果の精度は落ちる）。

## 参考文献

- SAA (Sample Average Approximation): Kleywegt et al., "The sample average approximation method for stochastic discrete optimization" (2002)
- Gaussian Copula: Nelsen, "An Introduction to Copulas" (2006)

## ライセンス

社内利用限定。
