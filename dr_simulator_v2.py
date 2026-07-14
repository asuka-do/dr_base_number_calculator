#!/usr/bin/env python3
"""
DRポートフォリオ最適化シミュレータ v2
────────────────────────────────────────────
① kW     : 離散N(5,2²)  [1,...,10] kW
② kWh    : 離散N(idx=1,σ=1) → [15,30,37.5,40,60,75] kWh
            kW大→kWh大 正相関（ガウスコピュラ ρ=0.75）
            フィルタ: kWh/kW > ⑦(3h) のみ採用
③ DR目標 : [100,250,500,750,1000,1250,1500] kW
④ 成功範囲: ③±10%
⑤ N* 算出: SAA+二分探索（1000kW, T=3h, 99.95%）
⑥ DR継続時間: 6h（グラフ用）
⑦ 最低持続時間: 3h
アクティブ条件: 全持続時間 > DR時刻t + ⑦
"""
import numpy as np, matplotlib, matplotlib.pyplot as plt, time, os
import japanize_matplotlib  # 日本語フォント対応
from matplotlib.lines import Line2D
from scipy.stats import norm as sp_norm
t0 = time.time()

# ── スタイル ───────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "figure.facecolor":"#0d1117","axes.facecolor":"#161b22",
    "axes.edgecolor":"#30363d","axes.labelcolor":"#c9d1d9",
    "text.color":"#c9d1d9","xtick.color":"#8b949e",
    "ytick.color":"#8b949e","grid.color":"#21262d",
    "legend.facecolor":"#161b22","legend.edgecolor":"#30363d",
})

# ══════════════════════════════════════════════════════════════════
#  CONFIG: パラメータ・定数定義（プログラム全体で使用）
# ══════════════════════════════════════════════════════════════════

# ── システム設定 ───────────────────────────────────────────────────
MC_SEED         = 42
POOL_SIZE       = 20_000
OUT             = "/mnt/user-data/outputs"
os.makedirs(OUT, exist_ok=True)   # 出力ディレクトリが無ければ作成

# ── 物理パラメータ ────────────────────────────────────────────────
SUCCESS_RANGE   = 0.10        # ④ DR成功幅 ±10%
MIN_DUR         = 3.0         # ⑦ 最低持続時間 [h]
T_BASE          = 3.0         # SAA基準のDR継続時間 [h]
T_FIG           = 6.0         # グラフ用のDR継続時間 [h]
KW_CORR         = 0.75        # ガウスコピュラ ρ（kW-kWh正相関）
FAILURE_RATE    = 0.06        # ⑧ 故障率 A_i ~ Bernoulli(1-FAILURE_RATE)
                              #    故障時(A_i=0)は kW寄与=0、持続時間=⑦扱い

# ── ① kW分布（離散正規）─────────────────────────────────────────
# kW値ラインナップを [KW_MIN, KW_MAX] の範囲・KW_STEP 刻みで生成し、
# 中央値 KW_MEDIAN・標準偏差 KW_STD の離散正規分布を作る
# ※ 変更例:
#    kWを 0.5kW刻みにしたい → KW_STEP=0.5
#    分散を大きくしたい     → KW_STD を大きく（例: 3.0）
#    中央値を変えたい       → KW_MEDIAN を変更
KW_MIN          = 1.0          # kW最小値
KW_MAX          = 10.0         # kW最大値
KW_STEP         = 1.0          # kW刻み幅
KW_MEDIAN       = 5.0          # 中央値 [kW]（正規なので median = 平均）
KW_STD          = 2.0          # 標準偏差 [kW]（分散 = KW_STD²）
KW_VALS         = np.arange(KW_MIN, KW_MAX + KW_STEP*0.5, KW_STEP)
kw_probs        = sp_norm.pdf(KW_VALS, KW_MEDIAN, KW_STD)
kw_probs       /= kw_probs.sum()

# ── ② kWh分布（離散正規、index空間で正規分布）──────────────────
# kWh値はラインナップ（不等間隔）で指定し、
# その index空間上で「中央値の位置」と「広がり」を制御する
# ※ 変更例:
#    ラインナップを変えたい → KWH_LINEUP を編集
#    中央値を変えたい       → KWH_MEDIAN を LINEUP 内の値に変更
#    分布を尖らせたい       → KWH_SPREAD を小さく（例: 0.5）
#    分布を平坦にしたい     → KWH_SPREAD を大きく（例: 2.0）
KWH_LINEUP      = [15., 30., 37.5, 40., 60., 75.]   # kWhラインナップ（昇順）
KWH_MEDIAN      = 30.0         # 中央値 [kWh]（必ず KWH_LINEUP 内の値）
KWH_SPREAD      = 1.0          # 標準偏差 (index空間)：1.0=隣接index間、2.0=広がる
KWH_VALS        = np.array(KWH_LINEUP)
assert KWH_MEDIAN in KWH_LINEUP, f"KWH_MEDIAN={KWH_MEDIAN} は KWH_LINEUP に含まれる値でなければなりません"
_kwh_med_idx    = KWH_LINEUP.index(KWH_MEDIAN)
kwh_probs       = sp_norm.pdf(np.arange(len(KWH_VALS)), _kwh_med_idx, KWH_SPREAD)
kwh_probs      /= kwh_probs.sum()

# ── ③ DR目標値・⑥ DR継続時間パターン ────────────────────────────
Q_TARGETS       = [100, 250, 500, 750, 1000, 1250, 1500]
Q_LABELS        = [f"{q}kW" for q in Q_TARGETS]
T_PATTERNS      = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
Q_NSTAR_REF     = 1000         # ⑤ N*算出基準のDR目標 [kW]

# ── ⑤ SAA + 二分探索（N*探索の試行数）─────────────────────────────
SUCCESS_PROB        = 0.9995   # N*要件: 達成確率 ≥ 99.95%
NT_NSTAR_EXPAND     = 300      # 上限探索フェーズ
NT_NSTAR_BISECT_S   = 1500     # 二分探索 (N ≤ 800)
NT_NSTAR_BISECT_M   = 800      # 二分探索 (800 < N ≤ 3000)
NT_NSTAR_BISECT_L   = 400      # 二分探索 (N > 3000)
NT_NSTAR_FAST       = 1500     # bootstrap内の各N*探索の試行数（高精度版）
NT_VERIFY           = 2000     # N* 達成率確認
NT_COUNT_FAIL       = 1000     # 失敗回数カウント
NSTAR_BOOTSTRAP_K   = 50       # N*(q)のσ評価：bootstrap反復数（高精度版）

# ── ⑥ 失敗確率テーブル(42パターン)の試行数 ───────────────────────
NT_TABLE_PROB       = 2000     # 各セルの達成率
NT_TABLE_COUNT      = 1000     # 各セルの失敗カウント

# ── MC試行数・時間解像度（グラフ用）────────────────────────────────
G_DUR           = 300          # Fig1 持続時間分布
G3              = 300          # Fig3 確率的版
N_TIME2         = 100          # Fig2 時間解像度
N_TIME3         = 120          # Fig3 時間解像度
N_TIME_F4       = 500          # Fig4 時間解像度

# ── RNGシードオフセット（再現性保証）──────────────────────────────
RNG_OFFSET_POOL = 0            # プール生成
RNG_OFFSET_MC   = 999          # SAA・N*探索・失敗確率テーブル
RNG_OFFSET_F1   = 11111        # Fig1 持続時間分布
RNG_OFFSET_G3   = 8888         # Fig3 確率的版

# ── プロット用定数 ─────────────────────────────────────────────────
BAR_WIDTH_F4    = 0.27         # Fig4 3カテゴリグループ棒の幅

# ── 色（kW=暖色系・kWh=寒色系で明確に区別）────────────────────────
KW_COLORS = [
    "#ff3838",  # 1kW  鮮紅
    "#ff5e1a",  # 2kW  赤橙
    "#ff8c00",  # 3kW  橙
    "#ffbe00",  # 4kW  琥珀
    "#e6e600",  # 5kW  黄
    "#9ee600",  # 6kW  黄緑
    "#3dbd3d",  # 7kW  緑
    "#ff1f8f",  # 8kW  ホットピンク
    "#ff6db0",  # 9kW  ライトピンク
    "#ff99cc",  # 10kW 淡ピンク
]
KWH_COLORS = [
    "#00c4ff",  # 15kWh  空色
    "#0077ee",  # 30kWh  青
    "#4422dd",  # 37.5kWh 藍
    "#7700cc",  # 40kWh  紫青
    "#aa00ee",  # 60kWh  紫
    "#dd44ff",  # 75kWh  薄紫
]

# ── プール生成（ガウスコピュラ・持続時間フィルタ）────────────────
def generate_pool(N_target, rng, corr=KW_CORR, min_dur=MIN_DUR):
    kw_cdf=np.cumsum(kw_probs); kwh_cdf=np.cumsum(kwh_probs)
    kw_acc,kwh_acc=[],[]
    batch=max(N_target*6,60000)
    while len(kw_acc)<N_target:
        L=np.array([[1.,0.],[corr,np.sqrt(max(1e-9,1-corr**2))]])
        Z=rng.standard_normal((batch,2))@L.T
        U=sp_norm.cdf(Z[:,0]); V=sp_norm.cdf(Z[:,1])
        ki=np.searchsorted(kw_cdf,U).clip(0,len(KW_VALS)-1)
        wi=np.searchsorted(kwh_cdf,V).clip(0,len(KWH_VALS)-1)
        kw=KW_VALS[ki]; kwh=KWH_VALS[wi]
        mask=kwh/kw>min_dur
        kw_acc.extend(kw[mask].tolist()); kwh_acc.extend(kwh[mask].tolist())
        if len(kw_acc)<N_target: batch=int(batch*1.5)
    kw_arr=np.array(kw_acc[:N_target]); kwh_arr=np.array(kwh_acc[:N_target])
    return kw_arr, kwh_arr, kwh_arr/kw_arr

print("プール生成中...")
rng_pool = np.random.default_rng(MC_SEED + RNG_OFFSET_POOL)
kw_pool, kwh_pool, dur_pool = generate_pool(POOL_SIZE, rng_pool)
print(f"  {POOL_SIZE:,}局生成完了（dur>{MIN_DUR}h でフィルタ済み）")
print(f"  ── 設定値 vs 実現値 ──")
print(f"    kW : 設定 中央値={KW_MEDIAN:.2f}, σ={KW_STD:.2f}")
print(f"         実現 中央値={np.median(kw_pool):.2f}, 平均={kw_pool.mean():.2f}, σ={kw_pool.std():.2f}")
print(f"    kWh: 設定 中央値={KWH_MEDIAN:.2f}, spread(idx)={KWH_SPREAD:.2f}")
print(f"         実現 中央値={np.median(kwh_pool):.2f}, 平均={kwh_pool.mean():.2f}, σ={kwh_pool.std():.2f}")
print(f"    dur: 実現 中央値={np.median(dur_pool):.2f}h, 平均={dur_pool.mean():.2f}h, min={dur_pool.min():.2f}h")

# ── SAA関数（アクティブ条件: dur > t + MIN_DUR、故障率 A_i 込み）─
def est_prob(N,T,lo_q,rng_mc,n_trials):
    idx=rng_mc.integers(0,POOL_SIZE,(n_trials,N))
    # A_i ~ Bernoulli(1-FAILURE_RATE): 1=稼働可能, 0=故障
    avail=rng_mc.random((n_trials,N))>=FAILURE_RATE
    eff=(kw_pool[idx]*(dur_pool[idx]>T+MIN_DUR)*avail).sum(1)
    return float((eff>=lo_q).mean())

def count_fail(N,T,lo_q,rng_mc,n_trials):
    idx=rng_mc.integers(0,POOL_SIZE,(n_trials,N))
    avail=rng_mc.random((n_trials,N))>=FAILURE_RATE
    eff=(kw_pool[idx]*(dur_pool[idx]>T+MIN_DUR)*avail).sum(1)
    return int((eff<lo_q).sum())

def mu_eff_at(T):
    # 期待値: E[kW × I(dur > T+⑦) × A] = (1-FAILURE_RATE) × E[kW × I(dur > T+⑦)]
    return float((kw_pool*(dur_pool>T+MIN_DUR)).mean()*(1-FAILURE_RATE))

MU_EFF={T:mu_eff_at(T) for T in T_PATTERNS+[T_BASE,T_FIG]}
print(f"  µ_eff(T=3h DR)={MU_EFF[T_BASE]:.3f}  µ_eff(T=6h DR)={MU_EFF[T_FIG]:.3f}")

# ── ⑤ N* 探索 ────────────────────────────────────────────────────
def find_n_star(q, rng_mc):
    """精密版: 二分探索で N*(q) を求める（合計約18,300試行）"""
    lo_q = q * (1 - SUCCESS_RANGE); mf = MU_EFF[T_BASE]
    if mf < 1e-6: return 9999
    lo_n = max(1, int(lo_q/mf*0.65)); hi_n = int(lo_q/mf*2.8) + 30
    for _ in range(38):
        if est_prob(hi_n, T_BASE, lo_q, rng_mc, NT_NSTAR_EXPAND) >= SUCCESS_PROB: break
        hi_n = int(hi_n*1.7) + 15
    while lo_n < hi_n:
        mid = (lo_n + hi_n) // 2
        nt = NT_NSTAR_BISECT_L if mid > 3000 else NT_NSTAR_BISECT_M if mid > 800 else NT_NSTAR_BISECT_S
        if est_prob(mid, T_BASE, lo_q, rng_mc, nt) >= SUCCESS_PROB: hi_n = mid
        else: lo_n = mid + 1
    return lo_n

def find_n_star_fast(q, rng_mc, n_trials=NT_NSTAR_FAST):
    """軽量版: 全フェーズ n_trials 試行で N*(q) を概算（Fig4表示用）"""
    lo_q = q * (1 - SUCCESS_RANGE); mf = MU_EFF[T_BASE]
    if mf < 1e-6: return 9999
    lo_n = max(1, int(lo_q/mf*0.65)); hi_n = int(lo_q/mf*2.8) + 30
    for _ in range(20):
        if est_prob(hi_n, T_BASE, lo_q, rng_mc, n_trials) >= SUCCESS_PROB: break
        hi_n = int(hi_n*1.7) + 15
    while lo_n < hi_n:
        mid = (lo_n + hi_n) // 2
        if est_prob(mid, T_BASE, lo_q, rng_mc, n_trials) >= SUCCESS_PROB: hi_n = mid
        else: lo_n = mid + 1
    return lo_n

rng_mc = np.random.default_rng(MC_SEED + RNG_OFFSET_MC)
print(f"\n⑤ N* 探索中 ({Q_NSTAR_REF}kW, T={T_BASE}h)...")
N_STAR = find_n_star(Q_NSTAR_REF, rng_mc)
lo_1000 = Q_NSTAR_REF * (1 - SUCCESS_RANGE)
p_star = est_prob(N_STAR, T_BASE, lo_1000, rng_mc, NT_VERIFY)
fc_star = count_fail(N_STAR, T_BASE, lo_1000, rng_mc, NT_COUNT_FAIL)
print(f"  N*({Q_NSTAR_REF}kW) = {N_STAR}局  達成={p_star*100:.2f}%  失敗={fc_star}/1000")

# ── N* の不確実性評価 ────────────────────────────────────────────
# find_n_star_fast を異なる seed で複数回実行し、N* 自体の σ を測る

print(f"\n  全DR目標のN*計算（{NSTAR_BOOTSTRAP_K}回bootstrap × 各{NT_NSTAR_FAST}試行）...")
print(f"  {'N* 平均':>10} ± {'σ':>4}  {'最小':>4}-{'最大':>4}")
N_stars = {}      # 後段で使う代表値（平均の整数値）
for q in Q_TARGETS:
    n_samples = np.zeros(NSTAR_BOOTSTRAP_K, dtype=int)
    for k in range(NSTAR_BOOTSTRAP_K):
        # 各試行ごとに異なる seed の RNG
        rng_k = np.random.default_rng(MC_SEED + RNG_OFFSET_MC + 7919*k + q)
        n_samples[k] = find_n_star_fast(q, rng_k, n_trials=NT_NSTAR_FAST)
    n_mean = n_samples.mean()
    n_std  = n_samples.std(ddof=1)   # 不偏分散
    n_min  = n_samples.min()
    n_max  = n_samples.max()
    N_stars[q] = int(round(n_mean))
    print(f"    N*({q:5d}kW) = {n_mean:6.1f} ± {n_std:4.1f}局   "
          f"({n_min:4d}-{n_max:4d})   採用={N_stars[q]}局")

# ── ⑥ 全36パターン失敗確率 ──────────────────────────────────────
print(f"\n⑥ 全{len(Q_TARGETS)*len(T_PATTERNS)}パターン計算中...")
fail_probs={}; fail_cnts={}
for q in Q_TARGETS:
    lo_q=q*(1-SUCCESS_RANGE)
    for T in T_PATTERNS:
        p = est_prob(N_STAR, T, lo_q, rng_mc, NT_TABLE_PROB)
        fc = count_fail(N_STAR, T, lo_q, rng_mc, NT_TABLE_COUNT)
        fail_probs[(q,T)]=1-p; fail_cnts[(q,T)]=fc

print(f"\n■ DR失敗確率 [%]  N*={N_STAR}局  アクティブ条件: dur > T + {MIN_DUR}h")
header=f"  {'③↓/⑥→':12s}|"+"".join(f"  {T:.1f}h   " for T in T_PATTERNS)
print(header)
for q in Q_TARGETS:
    row=f"  {q:5d}kW       |"
    for T in T_PATTERNS:
        fp=fail_probs[(q,T)]*100
        sym="○" if fp<1 else("△" if fp<50 else "×")
        row+=f" {sym}{fp:5.1f}%  "
    print(row)

# ══════════════════════════════════════════════════════════════════
#  Fig 1: フリート属性分布（4パネル）
# ══════════════════════════════════════════════════════════════════
fig,axes=plt.subplots(1,4,figsize=(26,6))
fig.suptitle(
    f"Fig1 v2: フリート属性分布  N*={N_STAR}局  故障率={FAILURE_RATE*100:.0f}%\n"
    f"kW~N({KW_MEDIAN},{KW_STD}²)  kWh∈{KWH_LINEUP}  dur>{MIN_DUR}h  ρ_copula={KW_CORR}",fontsize=12)

# ① kW
ax=axes[0]
kw_cnt = N_STAR*kw_probs; kw_std = np.sqrt(N_STAR*kw_probs*(1-kw_probs))
bars=ax.bar(KW_VALS,kw_cnt,width=0.82,color="#60a5fa",alpha=0.75,edgecolor="#1e40af",lw=0.8)
ax.errorbar(KW_VALS,kw_cnt,yerr=kw_std,fmt="none",color="white",capsize=5,capthick=1.5,elinewidth=1.5,alpha=0.88,label="±σ")
for bar,cnt,sd in zip(bars,kw_cnt,kw_std):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+sd+0.3,f"{cnt:.0f}±{sd:.1f}",ha="center",va="bottom",fontsize=7)
ax.axvline(KW_MEDIAN, color="#f59e0b", linestyle="--", lw=1.5, alpha=0.8, label=f"中央値={KW_MEDIAN:.0f}kW")
ax.set_title(f"① kW分布",fontsize=11); ax.set_xticks(KW_VALS); ax.legend(fontsize=7); ax.grid(True,alpha=0.3,axis="y")
ax.text(0.98,0.96,f"N={N_STAR}局",transform=ax.transAxes,fontsize=9,ha="right",va="top",color="#8b949e")

# ② kWh
ax=axes[1]
kwh_cnt = N_STAR*kwh_probs; kwh_std = np.sqrt(N_STAR*kwh_probs*(1-kwh_probs))
bars=ax.bar(range(len(KWH_VALS)),kwh_cnt,width=0.72,color="#34d399",alpha=0.75,edgecolor="#065f46",lw=0.8)
ax.errorbar(range(len(KWH_VALS)),kwh_cnt,yerr=kwh_std,fmt="none",color="white",capsize=5,capthick=1.5,elinewidth=1.5,alpha=0.88,label="±σ")
for i,(cnt,sd) in enumerate(zip(kwh_cnt,kwh_std)):
    ax.text(i,cnt+sd+0.3,f"{cnt:.0f}±{sd:.1f}",ha="center",va="bottom",fontsize=7)
ax.set_xticks(range(len(KWH_VALS))); ax.set_xticklabels([f"{v:.0f}" if v==int(v) else f"{v}" for v in KWH_VALS],fontsize=9)
ax.axvline(_kwh_med_idx, color="#f59e0b", linestyle="--", lw=1.5, alpha=0.8, label=f"中央値={KWH_MEDIAN:.0f}kWh")
ax.set_title("② kWh分布",fontsize=11); ax.legend(fontsize=7); ax.grid(True,alpha=0.3,axis="y")
ax.text(0.98,0.96,f"N={N_STAR}局",transform=ax.transAxes,fontsize=9,ha="right",va="top",color="#8b949e")

# 持続時間（確率的棒グラフ）
ax=axes[2]
rng_f1 = np.random.default_rng(MC_SEED + RNG_OFFSET_F1)
bins_dur=np.arange(MIN_DUR,31,1.); bin_c=(bins_dur[:-1]+bins_dur[1:])/2
hist_mat=np.zeros((G_DUR,len(bin_c)))
for trial in range(G_DUR):
    idx = rng_f1.integers(0, POOL_SIZE, N_STAR)
    counts,_=np.histogram(dur_pool[idx],bins=bins_dur)
    hist_mat[trial]=counts
dp25=np.percentile(hist_mat,25,axis=0); dp50=np.percentile(hist_mat,50,axis=0); dp75=np.percentile(hist_mat,75,axis=0)
dur_std=hist_mat.std(axis=0)
ax.bar(bin_c,dp50,width=0.82,color="#a78bfa",alpha=0.80,edgecolor="#4c1d95",lw=0.7)
ax.errorbar(bin_c,dp50,yerr=[dp50-dp25,dp75-dp50],fmt="none",color="white",capsize=4,capthick=1.2,elinewidth=1.2,alpha=0.85,label="25〜75%ile")
for x,h,hi,sd in zip(bin_c,dp50,dp75,dur_std):
    if h>=1: ax.text(x,hi+0.3,f"{h:.0f}±{sd:.1f}",ha="center",va="bottom",fontsize=6.5)
ax.axvline(T_BASE+MIN_DUR,color="#ef4444",linestyle="--",lw=2.,alpha=0.9,label=f"T_BASE+⑦={T_BASE+MIN_DUR:.0f}h")
ax.axvline(T_FIG+MIN_DUR, color="#f59e0b",linestyle="--",lw=1.5,alpha=0.8,label=f"T_FIG+⑦={T_FIG+MIN_DUR:.0f}h")
ax.set_xlabel("持続時間[h]"); ax.set_ylabel("局数")
ax.set_title(f"持続時間分布 確率的 {G_DUR}試行",fontsize=11)
ax.legend(fontsize=7); ax.grid(True,alpha=0.3,axis="y"); ax.set_xlim(MIN_DUR-0.5,30.5)
ax.text(0.98,0.96,f"N={N_STAR}局\n×{G_DUR}試行",transform=ax.transAxes,fontsize=9,ha="right",va="top",color="#8b949e")

# kW-kWh散布図（相関確認）
ax=axes[3]
sn=min(3000,POOL_SIZE); si=rng_f1.integers(0,POOL_SIZE,sn)
for i_kwh,(kwh_v,col) in enumerate(zip(KWH_VALS,KWH_COLORS)):
    mask=kwh_pool[si]==kwh_v
    if mask.sum()>0:
        xs=kw_pool[si][mask]+rng_f1.uniform(-0.12,0.12,mask.sum())
        ys=np.full(mask.sum(),kwh_v)+rng_f1.uniform(-0.5,0.5,mask.sum())
        ax.scatter(xs,ys,c=col,alpha=0.4,s=7,label=f"{kwh_v}kWh")
corr_val=np.corrcoef(kw_pool[si],kwh_pool[si])[0,1]
ax.set_title(f"kW-kWh相関散布図(N={sn:,})\nρ実測={corr_val:.3f}",fontsize=11)
ax.set_xlabel("kW"); ax.set_ylabel("kWh")
ax.set_xticks(KW_VALS); ax.set_yticks(KWH_VALS)
ax.set_yticklabels([f"{v:.0f}" if v==int(v) else f"{v}" for v in KWH_VALS])
ax.legend(fontsize=7,ncol=2); ax.grid(True,alpha=0.2)

plt.tight_layout()
plt.savefig(f"{OUT}/fig1v2_fleet_distribution.png",dpi=120,bbox_inches="tight",facecolor=fig.get_facecolor())
plt.close(); print("\nfig1v2_fleet_distribution.png 保存")

# ══════════════════════════════════════════════════════════════════
#  Fig 2 v2: ④達成の蓄電池kW・kWh組み合わせ（期待値版）
#  上段: kWカテゴリ別出力kW  下段: kWhカテゴリ別出力kW
# ══════════════════════════════════════════════════════════════════
T_arr = np.linspace(0, T_FIG, N_TIME2)

def det_composition(N_q,q,T_arr):
    """期待値ベースのkW・kWh別寄与を計算（故障率込み）"""
    lo_q=q*(1-SUCCESS_RANGE)
    avail_factor=1-FAILURE_RATE   # E[A_i] = 1-FAILURE_RATE
    kw_contribs =np.zeros((len(KW_VALS), len(T_arr)))
    kwh_contribs=np.zeros((len(KWH_VALS),len(T_arr)))
    for it,t in enumerate(T_arr):
        active=(dur_pool>t+MIN_DUR)
        # 期待容量 = N_q × E[kW × I(active) × A] = N_q × E[kW × I(active)] × (1-FAILURE_RATE)
        cap=(kw_pool*active).mean()*N_q*avail_factor
        scale=min(1.,q/cap) if cap>1e-9 else 0.
        for i_kw,kw_v in enumerate(KW_VALS):
            m=active&(kw_pool==kw_v)
            kw_contribs[i_kw,it]=(kw_pool*m).mean()*N_q*avail_factor*scale
        for i_kwh,kwh_v in enumerate(KWH_VALS):
            m=active&(kwh_pool==kwh_v)
            kwh_contribs[i_kwh,it]=(kw_pool*m).mean()*N_q*avail_factor*scale
    return kw_contribs,kwh_contribs

fig,axes=plt.subplots(4,4,figsize=(28,16),squeeze=False)
for r in range(4):
    for c in range(4): axes[r][c].set_visible(False)

fig.suptitle(
    f"Fig2 v2: ④達成の蓄電池kW・kWh組み合わせ（期待値版）  DR継続時間={T_FIG}h  N*={N_STAR}局  故障率={FAILURE_RATE*100:.0f}%\n"
    f"上段=kWカテゴリ別  下段=kWhカテゴリ別  アクティブ条件: 持続時間 > t + {MIN_DUR}h  期待値に (1-故障率) を乗算",fontsize=12,y=1.01)

for idx_q,(q,ql) in enumerate(zip(Q_TARGETS,Q_LABELS)):
    blk=idx_q//4; col=idx_q%4
    r_kw=blk*2; r_kwh=blk*2+1
    lo_q=q*(1-SUCCESS_RANGE); hi_q=q*(1+SUCCESS_RANGE)
    kw_c,kwh_c=det_composition(N_STAR,q,T_arr)
    total_kw=kw_c.sum(axis=0)

    for row,contribs,colors,label in [
        (r_kw, kw_c, KW_COLORS, [f"{int(v)}kW" for v in KW_VALS]),
        (r_kwh,kwh_c,KWH_COLORS,[f"{v:.0f}kWh" if v==int(v) else f"{v}kWh" for v in KWH_VALS])
    ]:
        ax=axes[row][col]; ax.set_visible(True)
        bottom=np.zeros(N_TIME2)
        # kWhは上から大きい順 → 小さい値から積んで大きい値を最上部に
        for i in range(len(colors)):
            ax.fill_between(T_arr,bottom,bottom+contribs[i],color=colors[i],alpha=0.82,linewidth=0)
            bottom+=contribs[i]
        # 成功ゾーン
        ax.axhline(lo_q,color="white",linestyle="--",lw=1.8,alpha=0.9)
        ax.axhline(hi_q,color="#fbbf24",linestyle="--",lw=1.2,alpha=0.8)
        ax.axhspan(0,lo_q,color="#ef4444",alpha=0.07)
        # 逸脱ライン（期待値が下限を下回る時刻）：線形補間で正確な交差点を算出
        sf=total_kw<lo_q
        if sf.any():
            idx=int(np.where(sf)[0][0])
            if idx>0:
                v0,v1=total_kw[idx-1],total_kw[idx]
                t_dev=float(T_arr[idx-1]+(lo_q-v0)/(v1-v0)*(T_arr[idx]-T_arr[idx-1])) if v1!=v0 else float(T_arr[idx])
            else:
                t_dev=float(T_arr[0])
            ax.axvline(t_dev,color="#ef4444",lw=2.,alpha=0.95)
            ax.text(t_dev+T_FIG*0.02,hi_q*1.02,f"逸脱\n{t_dev:.2f}h",ha="left",va="top",fontsize=8,color="#ef4444",
                    bbox=dict(boxstyle="round,pad=0.2",facecolor="#0d1117",alpha=0.8))
        title_sfx="kWカテゴリ別" if row==r_kw else "kWhカテゴリ別"
        ax.set_title(f"③={ql}  {title_sfx}",fontsize=10)
        ax.set_xlim(0,T_FIG); ax.set_ylim(0,hi_q*1.18)
        ax.set_xlabel("経過時間[h]",fontsize=9); ax.set_ylabel("出力kW",fontsize=9)
        ax.grid(True,alpha=0.15,lw=0.5)

# 凡例
kw_patches=[plt.Rectangle((0,0),1,1,color=KW_COLORS[i],alpha=0.82,label=f"{int(KW_VALS[i])}kW") for i in range(len(KW_VALS))]
kwh_patches=[plt.Rectangle((0,0),1,1,color=KWH_COLORS[i],alpha=0.82,label=f"{v:.0f}kWh" if v==int(v) else f"{v}kWh") for i,v in enumerate(KWH_VALS)]
extra=[Line2D([0],[0],color="white",lw=1.8,ls="--",label="lower(③×90%)"),
       Line2D([0],[0],color="#fbbf24",lw=1.2,ls="--",label="upper(③×110%)"),
       Line2D([0],[0],color="#ef4444",lw=2.,ls="-",label="期待値逸脱")]
plt.tight_layout(rect=[0,0.05,1,1])
leg1=fig.legend(handles=kw_patches,loc="lower left",bbox_to_anchor=(0.01,0.0),
                title="kWカテゴリ",ncol=5,fontsize=8,title_fontsize=8,framealpha=0.6)
leg2=fig.legend(handles=kwh_patches+extra,loc="lower right",bbox_to_anchor=(0.99,0.0),
                title="kWhカテゴリ / 凡例",ncol=5,fontsize=8,title_fontsize=8,framealpha=0.6)
fig.add_artist(leg1)
plt.savefig(f"{OUT}/fig2v2_composition_7panels.png",dpi=120,bbox_inches="tight",facecolor=fig.get_facecolor())
plt.close(); print("fig2v2_composition_7panels.png 保存")

# ══════════════════════════════════════════════════════════════════
#  Fig 3 v2: 確率的版（MC 300試行）
#  上段: kWカテゴリ別  下段: kWhカテゴリ別
#  右上: 逸脱ライン時刻の失敗率・失敗回数
# ══════════════════════════════════════════════════════════════════
T_plt3 = np.linspace(0, T_FIG, N_TIME3)

def mc_composition3(N_q,q,rng_g):
    lo_q=q*(1-SUCCESS_RANGE)
    idx=rng_g.integers(0,POOL_SIZE,(G3,N_q))
    kw_s=kw_pool[idx]; kwh_s=kwh_pool[idx]; dur_s=dur_pool[idx]
    # A_i ~ Bernoulli(1-FAILURE_RATE): 試行ごと、局ごとに故障判定
    avail=(rng_g.random((G3,N_q))>=FAILURE_RATE)
    kw_all =np.zeros((G3,len(KW_VALS), N_TIME3))
    kwh_all=np.zeros((G3,len(KWH_VALS),N_TIME3))
    for it,t in enumerate(T_plt3):
        active=(dur_s>t+MIN_DUR)&avail   # 故障局はアクティブから除外
        cap=(kw_s*active).sum(axis=1)
        scale=np.where(cap>1e-9,np.minimum(1.,q/cap),0.)
        for i_kw,kw_v in enumerate(KW_VALS):
            m=active&(kw_s==kw_v)
            kw_all[:,i_kw,it]=(kw_s*m).sum(axis=1)*scale
        for i_kwh,kwh_v in enumerate(KWH_VALS):
            m=active&(kwh_s==kwh_v)
            kwh_all[:,i_kwh,it]=(kw_s*m).sum(axis=1)*scale
    return kw_all,kwh_all

rng_g3 = np.random.default_rng(MC_SEED + RNG_OFFSET_G3)
fig,axes=plt.subplots(4,4,figsize=(28,16),squeeze=False)
for r in range(4):
    for c in range(4): axes[r][c].set_visible(False)

fig.suptitle(
    f"Fig3 v2: ④達成の蓄電池kW・kWh組み合わせ（確率的版 {G3}試行）  DR継続時間={T_FIG}h  N*={N_STAR}局  故障率={FAILURE_RATE*100:.0f}%\n"
    f"上段=kWカテゴリ別  下段=kWhカテゴリ別  白帯=25〜75%ile  右上=逸脱ライン時刻の失敗率  A_i~Bernoulli({(1-0.06):.2f})",fontsize=12,y=1.01)

for idx_q,(q,ql) in enumerate(zip(Q_TARGETS,Q_LABELS)):
    blk=idx_q//4; col=idx_q%4
    r_kw=blk*2; r_kwh=blk*2+1
    lo_q=q*(1-SUCCESS_RANGE); hi_q=q*(1+SUCCESS_RANGE)

    kw_all,kwh_all=mc_composition3(N_STAR,q,rng_g3)
    total_all=kw_all.sum(axis=1)   # (G3, N_TIME3)
    pcts=np.percentile(total_all,[10,25,50,75,90],axis=0)

    # 逸脱ライン判定：線形補間で正確な中央値の交差時刻を算出
    sf=pcts[2]<lo_q
    if sf.any():
        t_dev_idx=int(np.where(sf)[0][0])
        if t_dev_idx>0:
            v0,v1=pcts[2][t_dev_idx-1],pcts[2][t_dev_idx]
            t_dev=float(T_plt3[t_dev_idx-1]+(lo_q-v0)/(v1-v0)*(T_plt3[t_dev_idx]-T_plt3[t_dev_idx-1])) if v1!=v0 else float(T_plt3[t_dev_idx])
        else:
            t_dev=float(T_plt3[0])
        fail_mask=total_all[:,t_dev_idx]<lo_q-0.5
        t_dev_str=f"@{t_dev:.2f}h"
    else:
        t_dev=None; fail_mask=np.zeros(G3,dtype=bool); t_dev_str="(逸脱なし)"

    fp_mc=fail_mask.mean()*100; fc_mc=int(fail_mask.sum())

    for row,all_data,colors in [
        (r_kw, kw_all, KW_COLORS),
        (r_kwh,kwh_all,KWH_COLORS)
    ]:
        ax=axes[row][col]; ax.set_visible(True)
        medians=np.median(all_data,axis=0)
        bottom=np.zeros(N_TIME3)
        # kWhは上から大きい順 → 小さい値から積んで大きい値を最上部に
        for i in range(len(colors)):
            ax.fill_between(T_plt3,bottom,bottom+medians[i],color=colors[i],alpha=0.78,linewidth=0)
            bottom+=medians[i]
        # 確率的バンド（合計出力）
        ax.fill_between(T_plt3,pcts[0],pcts[4],color="white",alpha=0.08,zorder=5)
        ax.fill_between(T_plt3,pcts[1],pcts[3],color="white",alpha=0.20,zorder=6)
        ax.plot(T_plt3,pcts[2],"w-",lw=2.,alpha=0.88,zorder=7)
        # ゾーン
        ax.axhline(lo_q,color="white",linestyle="--",lw=1.8,alpha=0.9)
        ax.axhline(hi_q,color="#fbbf24",linestyle="--",lw=1.2,alpha=0.8)
        ax.axhspan(0,lo_q,color="#ef4444",alpha=0.07)
        # 逸脱ライン
        if t_dev is not None:
            ax.axvline(t_dev,color="#ef4444",lw=2.,alpha=0.95,zorder=9)
            ax.text(t_dev+T_FIG*0.02,hi_q*1.02,f"逸脱\n{t_dev:.2f}h",ha="left",va="top",fontsize=8,
                    color="#ef4444",bbox=dict(boxstyle="round,pad=0.2",facecolor="#0d1117",alpha=0.8))
        # est/cnt（上段のみ右上に表示）
        if row==r_kw:
            scl="#34d399" if fp_mc<1 else("#f59e0b" if fp_mc<50 else "#ef4444")
            sym="○" if fp_mc<1 else("△" if fp_mc<50 else "×")
            ax.text(0.98,0.98,f"{sym} {fc_mc}/{G3}試行\n{t_dev_str}",
                    transform=ax.transAxes,fontsize=8.5,ha="right",va="top",color=scl,
                    bbox=dict(boxstyle="round,pad=0.35",facecolor="#0d1117",alpha=0.85),zorder=10)
        title_sfx="kWカテゴリ別" if row==r_kw else "kWhカテゴリ別"
        ax.set_title(f"③={ql}  {title_sfx}",fontsize=10)
        ax.set_xlim(0,T_FIG); ax.set_ylim(0,hi_q*1.18)
        ax.set_xlabel("経過時間[h]",fontsize=9); ax.set_ylabel("出力kW",fontsize=9)
        ax.grid(True,alpha=0.15,lw=0.5)

# 凡例
kw_patches=[plt.Rectangle((0,0),1,1,color=KW_COLORS[i],alpha=0.82,label=f"{int(KW_VALS[i])}kW") for i in range(len(KW_VALS))]
kwh_patches=[plt.Rectangle((0,0),1,1,color=KWH_COLORS[i],alpha=0.82,label=f"{v:.0f}kWh" if v==int(v) else f"{v}kWh") for i,v in enumerate(KWH_VALS)]
extra2=[Line2D([0],[0],color="white",lw=2.,ls="-",label="中央値(50%ile)"),
        Line2D([0],[0],color="white",lw=5,ls="-",alpha=0.20,label="25〜75%ile"),
        Line2D([0],[0],color="white",lw=3,ls="-",alpha=0.08,label="10〜90%ile"),
        Line2D([0],[0],color="white",lw=1.8,ls="--",label="lower(③×90%)"),
        Line2D([0],[0],color="#fbbf24",lw=1.2,ls="--",label="upper(③×110%)"),
        Line2D([0],[0],color="#ef4444",lw=2.,ls="-",label="中央値逸脱")]
plt.tight_layout(rect=[0,0.05,1,1])
leg1=fig.legend(handles=kw_patches,loc="lower left",bbox_to_anchor=(0.01,0.0),
                title="kWカテゴリ",ncol=5,fontsize=8,title_fontsize=8,framealpha=0.6)
leg2=fig.legend(handles=kwh_patches+extra2,loc="lower right",bbox_to_anchor=(0.99,0.0),
                title="kWhカテゴリ / 凡例",ncol=6,fontsize=8,title_fontsize=8,framealpha=0.6)
fig.add_artist(leg1)
plt.savefig(f"{OUT}/fig3v2_composition_stochastic_7panels.png",dpi=120,bbox_inches="tight",facecolor=fig.get_facecolor())
plt.close(); print("fig3v2_composition_stochastic_7panels.png 保存")


# ══════════════════════════════════════════════════════════════════
#  Fig 4 v2: 683局のうち稼働数・退場数・未使用数（kWh別）
#
#  N*=683局 固定（1000kW/3h算出値）
#  DR目標q ごとに N*(q) を計算し:
#     effective_used = min(683, N*(q))   ← DRに使用された局数
#     未使用数       = 683 - effective_used  ← 余剰のため一度も使用されない局
#  effective_used のうち:
#     稼働数 = 逸脱時点（または T=6h）で稼働中
#     退場数 = 持続時間⑦到達で退場済み
#  合計 = 稼働 + 退場 + 未使用 = 683
# ══════════════════════════════════════════════════════════════════
T_arr_f4 = np.linspace(0, T_FIG, N_TIME_F4)

def get_stats_3way(q):
    """683局を稼働/退場/未使用に3分類"""
    lo_q = q * (1 - SUCCESS_RANGE)
    N_q  = N_stars[q]                        # 目標別N*
    effective_used = min(N_STAR, N_q)        # 実際に使用する局数 (683キャップ)
    unused_count   = N_STAR - effective_used # 余剰: 未使用

    # 逸脱時刻 t_dev（N*=683局で計算 → Fig2/3と完全一致、故障率込み）
    avail_factor=1-FAILURE_RATE
    raw      = np.array([(kw_pool*(dur_pool>t+MIN_DUR)).mean()*N_STAR*avail_factor for t in T_arr_f4])
    total_kw = np.minimum(raw, q)
    sf = total_kw < lo_q
    if sf.any():
        idx = int(np.where(sf)[0][0])
        if idx > 0:
            v0,v1 = total_kw[idx-1], total_kw[idx]
            td = float(T_arr_f4[idx-1]+(lo_q-v0)/(v1-v0)*(T_arr_f4[idx]-T_arr_f4[idx-1])) if v1!=v0 else float(T_arr_f4[idx])
        else:
            td = float(T_arr_f4[0])
        # 評価時刻 = 逸脱ラインの「次の離散時間点」 = T_arr_f4[idx]
        t_eval = float(T_arr_f4[idx])
        active_m  = dur_pool >  t_eval + MIN_DUR    # 持続時間⑦以上残っている
        retired_m = dur_pool <= t_eval + MIN_DUR    # 持続時間⑦に到達
    else:
        td = None
        t_eval = None
        # 逸脱なし → 持続時間で見れば全局稼働中（ただし故障局は別途退場扱い）
        active_m  = np.ones(POOL_SIZE, dtype=bool)
        retired_m = np.zeros(POOL_SIZE, dtype=bool)

    # kWh別の3カテゴリ（故障率を反映）
    # 稼働: 使用×(1-故障率)×P(kWh=kh & 持続時間⑦以上)
    # 退場: 使用×{故障率×P(kWh=kh) + (1-故障率)×P(kWh=kh & 持続時間⑦到達)}
    # 未使用: 余剰×P(kWh=kh)
    p_fail=FAILURE_RATE
    ac_kwh = np.array([((kwh_pool==kh)&active_m ).mean()*effective_used*(1-p_fail) for kh in KWH_VALS])
    ex_kwh = np.array([((kwh_pool==kh)).mean()*effective_used*p_fail
                       + ((kwh_pool==kh)&retired_m).mean()*effective_used*(1-p_fail) for kh in KWH_VALS])
    un_kwh = np.array([(kwh_pool==kh).mean()*unused_count for kh in KWH_VALS])

    # 平均と分散（kWhの統計）
    kwh_ac = kwh_pool[active_m]
    kwh_ex = kwh_pool[retired_m]
    m_ac = float(kwh_ac.mean()) if len(kwh_ac) else 0.
    v_ac = float(kwh_ac.var())  if len(kwh_ac) else 0.
    m_ex = float(kwh_ex.mean()) if len(kwh_ex) else 0.
    v_ex = float(kwh_ex.var())  if len(kwh_ex) else 0.

    n_ac = ac_kwh.sum()
    n_ex_calc = ex_kwh.sum()
    n_ac = float(n_ac)
    n_ex = float(n_ex_calc)
    return td, t_eval, ac_kwh, ex_kwh, un_kwh, m_ac, v_ac, m_ex, v_ex, n_ac, n_ex, unused_count, effective_used, N_q

print(f"\nFig4: N*={N_STAR}局を稼働/退場/未使用に3分類して計算中...")
fig4, axes4 = plt.subplots(2, 4, figsize=(28, 10), squeeze=False)
for c in range(4): axes4[1][c].set_visible(False)

fig4.suptitle(
    f"Fig4 v2: N*={N_STAR}局を「稼働」「退場」「未使用」に3分類（kWh別）  故障率={FAILURE_RATE*100:.0f}%\n"
    f"横軸=kWhラインナップ  緑=稼働数（働ける×持続時間⑦以上）  赤=退場数（故障 OR 持続時間⑦到達）  灰=未使用数\n"
    f"稼働+退場+未使用={N_STAR}局（固定）",
    fontsize=12, y=1.03)

W = BAR_WIDTH_F4
x_kwh = np.arange(len(KWH_VALS))
kwh_labels = [f"{v:.0f}" if v==int(v) else f"{v}" for v in KWH_VALS]

for idx_q,(q,ql) in enumerate(zip(Q_TARGETS,Q_LABELS)):
    r = idx_q//4; c = idx_q%4
    ax = axes4[r][c]; ax.set_visible(True)
    td,t_eval,ac_kwh,ex_kwh,un_kwh,m_ac,v_ac,m_ex,v_ex,n_ac,n_ex,n_un,n_eff,N_q = get_stats_3way(q)
    if td is not None:
        title_sfx = f"逸脱 t={td:.3f}h → 評価 t={t_eval:.3f}h（次の離散点）"
    else:
        title_sfx = f"逸脱なし（退場数=0）"

    # 3本の棒グラフ
    ax.bar(x_kwh-W, ac_kwh, width=W, color="#34d399", alpha=0.88,
           edgecolor="#065f46", linewidth=0.6, label=f"稼働数 ({n_ac:.0f}局)")
    ax.bar(x_kwh,   ex_kwh, width=W, color="#f87171", alpha=0.78,
           edgecolor="#7f1d1d", linewidth=0.6, label=f"退場数 ({n_ex:.0f}局)")
    ax.bar(x_kwh+W, un_kwh, width=W, color="#94a3b8", alpha=0.65,
           edgecolor="#475569", linewidth=0.6, label=f"未使用数 ({n_un:.0f}局)")

    # 棒の上に数値
    for x,h in zip(x_kwh, ac_kwh):
        if h>=0.3: ax.text(x-W, h+0.5, f"{h:.0f}", ha="center", va="bottom", fontsize=6.5, color="#34d399")
    for x,h in zip(x_kwh, ex_kwh):
        if h>=0.3: ax.text(x,   h+0.5, f"{h:.0f}", ha="center", va="bottom", fontsize=6.5, color="#f87171")
    for x,h in zip(x_kwh, un_kwh):
        if h>=0.3: ax.text(x+W, h+0.5, f"{h:.0f}", ha="center", va="bottom", fontsize=6.5, color="#94a3b8")

    # 平均と分散はタイトル末尾に簡潔に統合
    stats = (f"稼働 平均={m_ac:.1f}kWh σ²={v_ac:.1f}  "
             f"退場 平均={m_ex:.1f}kWh σ²={v_ex:.1f}")
    ax.set_title(f"③={ql}  N*(q)={N_q}局  {title_sfx}\n{stats}", fontsize=9)
    ax.set_xlabel("kWhラインナップ [kWh]", fontsize=9)
    ax.set_ylabel("期待局数", fontsize=9)
    ax.set_xticks(x_kwh); ax.set_xticklabels(kwh_labels, fontsize=9)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.grid(True, alpha=0.2, axis="y")

plt.tight_layout()
plt.savefig(f"{OUT}/fig4v2_deviation_combo.png", dpi=120, bbox_inches="tight",
            facecolor=fig4.get_facecolor())
plt.close(); print("fig4v2_deviation_combo.png 保存")

elapsed=time.time()-t0
print(f"\n{'━'*55}")
print(f"  全グラフ保存完了 (計算時間: {elapsed:.1f}秒)")
print(f"  fig1v2: kW/kWh/持続時間分布 + kW-kWh相関散布図")
print(f"  fig2v2: 期待値版 kW・kWh組み合わせ (T={T_FIG}h, 7パターン)")
print(f"  fig3v2: 確率的版 kW・kWh組み合わせ (T={T_FIG}h, {G3}試行)")
print(f"  fig4v2: 逸脱ライン時点の参加蓄電池 kW×kWh組み合わせ")
print(f"{'━'*55}")
