import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize
from scipy import stats
from itertools import combinations

try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Международная диверсификация портфеля",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background: #0a0e1a;
    color: #e0e6f0;
}

h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }

.metric-card {
    background: linear-gradient(135deg, #111827 0%, #1a2235 100%);
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin: 0.4rem 0;
}
.metric-label {
    font-size: 0.72rem;
    color: #6b8cba;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-family: 'IBM Plex Mono', monospace;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 0.2rem;
}
.positive { color: #34d399; }
.negative { color: #f87171; }
.neutral  { color: #60a5fa; }

.section-header {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #3b82f6;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 0.5rem;
    margin: 1.5rem 0 1rem;
}

.insight-box {
    background: #0f1f35;
    border-left: 3px solid #3b82f6;
    border-radius: 0 6px 6px 0;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    line-height: 1.6;
    color: #cbd5e1;
}
.insight-box strong { color: #93c5fd; }

.stMultiSelect [data-baseweb="tag"] {
    background-color: #1e3a5f !important;
    color: #93c5fd !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Index Universe ─────────────────────────────────────────────────────────────
INDEX_UNIVERSE = {
    # США
    "S&P 500 (США)":          "^GSPC",
    "NASDAQ 100 (США)":       "^NDX",
    "Dow Jones (США)":        "^DJI",
    "Russell 2000 (США)":     "^RUT",
    # Европа
    "Euro Stoxx 50 (ЕС)":     "^STOXX50E",
    "DAX (Германия)":         "^GDAXI",
    "FTSE 100 (Великобритания)": "^FTSE",
    "CAC 40 (Франция)":       "^FCHI",
    # Азия
    "Nikkei 225 (Япония)":    "^N225",
    "Hang Seng (Гонконг)":    "^HSI",
    "CSI 300 (Китай)":        "000300.SS",
    "Kospi (Южная Корея)":    "^KS11",
    "ASX 200 (Австралия)":    "^AXJO",
    # Развивающиеся рынки
    "Bovespa (Бразилия)":     "^BVSP",
    "NSE Nifty 50 (Индия)":   "^NSEI",
    "IPC (Мексика)":          "^MXX",
    "JSE (ЮАР)":              "^J203.JO",
    "MOEX (Россия)":          "IMOEX.ME",
    # Товары / альтернативы
    "Gold ETF (Золото)":      "GLD",
    "Oil ETF (Нефть)":        "USO",
    # DXY всегда добавляется отдельно
}

DXY_TICKER = "DX-Y.NYB"

# DEFAULT_SELECTION = [
#     "S&P 500 (США)",
#     "Euro Stoxx 50 (ЕС)",
#     "Nikkei 225 (Япония)",
#     "NSE Nifty 50 (Индия)",
# ]

DEFAULT_SELECTION = [
    "NASDAQ 100 (США)",
    "NSE Nifty 50 (Индия)",
    "Bovespa (Бразилия)",
    "Kospi (Южная Корея)"
]

COLORS = [
    "#3b82f6", "#34d399", "#f59e0b", "#f87171",
    "#a78bfa", "#38bdf8", "#fb7185", "#86efac",
    "#fbbf24", "#c084fc", "#22d3ee", "#4ade80",
]

# ─── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def load_data(tickers: list, start: str, end: str):
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw
    prices = prices.dropna(how="all")
    return prices

# Загрузка всех индексов universe для оптимального портфеля
@st.cache_data(ttl=3600)
def load_universe_returns(start: str, end: str):
    all_universe_tickers = list(INDEX_UNIVERSE.values())
    raw = yf.download(all_universe_tickers, start=start, end=end,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices_u = raw["Close"]
    else:
        prices_u = raw
    prices_u = prices_u.dropna(how="all").ffill()
    # Rename tickers → names
    ticker_to_name = {v: k for k, v in INDEX_UNIVERSE.items()}
    prices_u = prices_u.rename(columns=ticker_to_name)
    # Drop columns with too much missing data (>30%)
    thresh = int(len(prices_u) * 0.7)
    prices_u = prices_u.dropna(axis=1, thresh=thresh)
    return prices_u.pct_change().dropna(how="all")


def annual_return(ret_series):
    total = (1 + ret_series).prod()
    n_years = len(ret_series) / 252
    return total ** (1 / n_years) - 1 if n_years > 0 else 0

def annual_vol(ret_series):
    return ret_series.std() * np.sqrt(252)

def sharpe(ret_series, rf=0.0):
    ar = annual_return(ret_series)
    av = annual_vol(ret_series)
    return (ar - rf) / av if av > 0 else 0

def max_drawdown(price_series):
    roll_max = price_series.cummax()
    dd = (price_series - roll_max) / roll_max
    return dd.min()

def build_equal_weight_portfolio(returns_df):
    w = np.ones(returns_df.shape[1]) / returns_df.shape[1]
    port_ret = returns_df.dot(w)
    return port_ret, w

def corr_weight_portfolio(returns_df):
    corr = returns_df.corr()
    avg_corr = corr.mean()
    inv = 1 / avg_corr
    w = inv / inv.sum()
    port_ret = returns_df.dot(w.values)
    return port_ret, w.values

def color_corr(val):
    if pd.isna(val):
        return "#1a2235"
    if val >= 0.8:
        return "#7f1d1d"
    elif val >= 0.5:
        return "#92400e"
    elif val >= 0.2:
        return "#1e3a5f"
    elif val >= -0.2:
        return "#1a2235"
    else:
        return "#064e3b"


def max_sharpe_portfolio(returns_df, rf=0.0):
    n = returns_df.shape[1]
    mean_ret = returns_df.mean() * 252
    cov = returns_df.cov() * 252

    def neg_sharpe(w):
        w = np.array(w)
        r = w @ mean_ret
        v = np.sqrt(w @ cov.values @ w)
        return -(r - rf) / v if v > 0 else 0

    constraints = [{"type": "eq", "func": lambda w: np.sum(w) - 1}]  # исправь на "fun"
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n
    res = minimize(neg_sharpe, w0, method="SLSQP",
                   bounds=bounds, constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}])
    w_opt = res.x if res.success else w0
    port_ret = returns_df.dot(w_opt)
    return port_ret, w_opt

def var_cvar(ret_series, confidence=0.95):
    """Value at Risk и CVaR (Expected Shortfall) на дневном горизонте"""
    sorted_ret = ret_series.sort_values()
    idx = int((1 - confidence) * len(sorted_ret))
    var = sorted_ret.iloc[idx]           # квантиль
    cvar = sorted_ret.iloc[:idx].mean()  # среднее хуже VaR
    return var, cvar

def min_variance_portfolio(returns_df):
    n = returns_df.shape[1]
    cov = returns_df.cov() * 252
    
    def port_vol(w):
        return np.sqrt(w @ cov.values @ w)
    
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n
    res = minimize(port_vol, w0, method="SLSQP",
                   bounds=bounds, constraints=constraints)
    w_opt = res.x if res.success else w0
    return w_opt


def compute_efficient_frontier(returns_df, rf=0.0, n_points=400):
    """
    Возвращает:
      - frontier_vols, frontier_rets  — координаты кривой
      - rand_vols, rand_rets          — облако случайных портфелей
      - w_sharpe                      — веса макс. Sharpe
      - w_minvar                      — веса мин. дисперсии
    """
    n = returns_df.shape[1]
    mean_ret = returns_df.mean() * 252
    cov      = returns_df.cov()  * 252

    # Случайные портфели
    np.random.seed(42)
    n_rand = 3000
    rand_w    = np.random.dirichlet(np.ones(n), size=n_rand)
    rand_rets = rand_w @ mean_ret.values
    rand_vols = np.sqrt(np.einsum('ij,jk,ik->i', rand_w, cov.values, rand_w))

    # Граница эффективности: минимизируем дисперсию при заданном уровне доходности
    target_min = mean_ret.min()
    target_max = mean_ret.max()
    targets = np.linspace(target_min, target_max, n_points)

    frontier_vols, frontier_rets = [], []
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n

    for target in targets:
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: w @ mean_ret.values - t},
        ]
        res = minimize(
            lambda w: np.sqrt(w @ cov.values @ w),
            w0, method="SLSQP", bounds=bounds, constraints=constraints,
        )
        if res.success:
            frontier_vols.append(res.fun)
            frontier_rets.append(target)

    # Макс. Sharpe
    def neg_sharpe(w):
        r = w @ mean_ret.values
        v = np.sqrt(w @ cov.values @ w)
        return -(r - rf) / v if v > 0 else 0

    res_sh = minimize(neg_sharpe, w0, method="SLSQP",
                      bounds=bounds,
                      constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}])
    w_sharpe = res_sh.x if res_sh.success else w0

    # Мин. дисперсия
    w_minvar = min_variance_portfolio(returns_df)

    return (np.array(frontier_vols), np.array(frontier_rets),
            rand_vols, rand_rets, w_sharpe, w_minvar)


def find_best_portfolio_comp(returns_df, rf=0.0, max_assets=4, n_opt=300):
    """
    Перебирает все комбинации до max_assets индексов,
    для каждой оптимизирует веса на макс. Sharpe.
    Возвращает топ-10 результатов.
    """
    cols = returns_df.columns.tolist()
    results = []

    # for n_assets in range(2, max_assets + 1):
    n_assets = max_assets
    for combo in combinations(cols, n_assets):
        sub = returns_df[list(combo)].dropna()
        if len(sub) < 60:
            continue
        mean_r = sub.mean() * 252
        cov    = sub.cov() * 252
        n      = len(combo)

        def neg_sharpe(w):
            r = w @ mean_r.values
            v = np.sqrt(w @ cov.values @ w)
            return -(r - rf) / v if v > 0 else 0

        best_sh, best_w = np.inf, np.ones(n) / n
        # Multiple random starts to avoid local minima
        for _ in range(n_opt):
            w0 = np.random.dirichlet(np.ones(n))
            res = minimize(neg_sharpe, w0, method="SLSQP",
                           bounds=[(0.0, 1.0)] * n,
                           constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1}])
            if res.success and res.fun < best_sh:
                best_sh, best_w = res.fun, res.x
        port_r = sub.dot(best_w)
        results.append({
            "assets":    list(combo),
            "weights":   best_w,
            "sharpe":    -best_sh,
            "cagr":      annual_return(port_r) * 100,
            "vol":       annual_vol(port_r) * 100,
            "max_dd":    max_drawdown((1 + port_r).cumprod()) * 100,
            "returns":   port_r,
        })

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    return results[:10]

def find_best_portfolio(returns_df, rf=0.0, max_assets=4, n_opt=0):
    """
    Аналитическое решение: w* = Σ⁻¹·(μ−rf), нормализованное.
    n_opt оставлен для совместимости интерфейса, не используется.
    Время: <1 сек для любого числа индексов.
    """
    cols     = returns_df.columns.tolist()
    mean_all = returns_df.mean() * 252
    cov_all  = returns_df.cov()  * 252
    results  = []

    for n_assets in range(2, max_assets + 1):
        for combo in combinations(cols, n_assets):
            combo  = list(combo)
            mu_c   = mean_all[combo].values
            cov_c  = cov_all.loc[combo, combo].values

            try:
                excess = mu_c - rf
                # Решаем Σ·z = (μ−rf) — numerically stable vs явного инвертирования
                z = np.linalg.solve(
                    cov_c + np.eye(n_assets) * 1e-8,   # регуляризация
                    excess
                )
                w = np.clip(z, 0, None)   # long-only: отрицательные → 0
                if w.sum() < 1e-10:
                    w = np.ones(n_assets) / n_assets   # fallback: равные веса
                else:
                    w /= w.sum()
            except np.linalg.LinAlgError:
                w = np.ones(n_assets) / n_assets

            sub      = returns_df[combo].dropna()
            port_ret = sub.dot(w)
            sh       = sharpe(port_ret, rf)

            results.append({
                "assets":  combo,
                "weights": w,
                "sharpe":  sh,
                "cagr":    annual_return(port_ret) * 100,
                "vol":     annual_vol(port_ret) * 100,
                "max_dd":  max_drawdown((1 + port_ret).cumprod()) * 100,
                "returns": port_ret,
            })

    results.sort(key=lambda x: x["sharpe"], reverse=True)
    return results[:10]

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 Настройки")
    st.markdown("---")

    selected_names = st.multiselect(
        "Выберите индексы (4–8 рекомендуется)",
        options=list(INDEX_UNIVERSE.keys()),
        default=DEFAULT_SELECTION,
    )

    st.markdown("---")
    years = st.slider("Период анализа (лет)", min_value=3, max_value=15, value=10)
    include_dxy = st.checkbox("Включить DXY (Индекс доллара)", value=True)

    weight_method = st.radio(
        "Метод взвешивания портфеля",
        ["Равные веса (1/N)", "Веса по корреляции (обратные)", "Максимальный Sharpe (оптимизация)"],
    )

    rf_rate = st.number_input("Безрисковая ставка (%)", value=4.0, step=0.5) / 100

    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.72rem;color:#6b8cba;font-family:IBM Plex Mono'>© Портфельный анализ<br>данные: Yahoo Finance</div>",
        unsafe_allow_html=True,
    )

# ─── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# 🌍 Международная диверсификация портфеля")
st.markdown(
    "<div style='color:#6b8cba;font-size:0.9rem;margin-bottom:1.5rem'>"
    "Корреляционный анализ мировых индексов · Оптимизация портфеля · Влияние DXY"
    "</div>",
    unsafe_allow_html=True,
)

if len(selected_names) < 2:
    st.warning("Пожалуйста, выберите хотя бы 2 индекса в боковой панели.")
    st.stop()

# ─── Load Data ─────────────────────────────────────────────────────────────────
end_date   = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")

selected_tickers = {n: INDEX_UNIVERSE[n] for n in selected_names}
all_tickers = list(selected_tickers.values())
if include_dxy:
    all_tickers.append(DXY_TICKER)

with st.spinner("Загрузка данных с Yahoo Finance…"):
    prices_raw = load_data(all_tickers, start_date, end_date)

# Rename columns
rename_map = {v: k for k, v in INDEX_UNIVERSE.items()}
rename_map[DXY_TICKER] = "DXY (Индекс $)"
prices = prices_raw.rename(columns=rename_map)

# Keep only selected + DXY
keep_cols = selected_names.copy()
if include_dxy and "DXY (Индекс $)" in prices.columns:
    keep_cols.append("DXY (Индекс $)")
prices = prices[[c for c in keep_cols if c in prices.columns]].dropna(how="all").ffill()

returns = prices.pct_change().dropna()

with st.spinner("Загрузка всех индексов universe…"):
    universe_returns = load_universe_returns(start_date, end_date)
universe_cols = universe_returns.columns.tolist()

index_cols   = [c for c in selected_names if c in prices.columns]
dxy_col      = "DXY (Индекс $)" if ("DXY (Индекс $)" in prices.columns and include_dxy) else None
index_returns = returns[index_cols]

# ─── Tab Layout ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab_ef, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Динамика индексов",
    "🔗 Корреляционный анализ",
    "💼 Портфельный анализ",
    "Efficient Frontier",
    "📊 Сводная статистика",
    "📡 Rolling Beta",
    "🎲 Стохастические модели",
    "🏆 Оптимальный портфель"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Динамика
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Нормализованная динамика (база = 100)</div>', unsafe_allow_html=True)

    norm = prices[index_cols].div(prices[index_cols].iloc[1]) * 100
    # print(index_cols)
    # print(norm['Kospi (Южная Корея)'])
    # print(prices['Kospi (Южная Корея)'])

    fig = go.Figure()
    for i, col in enumerate(index_cols):
        fig.add_trace(go.Scatter(
            x=norm.index, y=norm[col],
            name=col, line=dict(color=COLORS[i % len(COLORS)], width=1.8),
            hovertemplate="%{y:.1f}<extra>" + col + "</extra>",
        ))

    if dxy_col:
        norm_dxy = prices[dxy_col] / prices[dxy_col].iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=norm_dxy.index, y=norm_dxy,
            name="DXY (Индекс $)",
            line=dict(color="#fbbf24", width=1.5, dash="dot"),
            hovertemplate="%{y:.1f}<extra>DXY</extra>",
        ))

    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
        height=440, legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45"),
        margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Annual returns bar
    st.markdown('<div class="section-header">Годовая доходность по индексам</div>', unsafe_allow_html=True)
    ann_rets = {col: annual_return(index_returns[col]) * 100 for col in index_cols}
    colors_bar = ["#34d399" if v >= 0 else "#f87171" for v in ann_rets.values()]
    fig2 = go.Figure(go.Bar(
        x=list(ann_rets.keys()), y=list(ann_rets.values()),
        marker_color=colors_bar,
        text=[f"{v:.1f}%" for v in ann_rets.values()],
        textposition="outside",
    ))
    fig2.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
        height=320, yaxis_title="% годовых (CAGR)",
        xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45"),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Корреляция
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Матрица попарной корреляции</div>', unsafe_allow_html=True)

    corr_cols = index_cols + ([dxy_col] if dxy_col else [])
    corr_matrix = returns[corr_cols].corr()

    # Heatmap
    fig_corr = go.Figure(go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale=[
            [0.0,  "#064e3b"],
            [0.35, "#1e3a5f"],
            [0.5,  "#1a2235"],
            [0.65, "#92400e"],
            [1.0,  "#7f1d1d"],
        ],
        zmid=0, zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=12, family="IBM Plex Mono"),
        hoverongaps=False,
        colorbar=dict(title="ρ", tickfont=dict(color="#e0e6f0")),
    ))
    fig_corr.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
        height=420, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(tickangle=-35),
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # Rolling correlation (first index vs others)
    st.markdown('<div class="section-header">Скользящая корреляция с ' + index_cols[0] + ' (252 дня)</div>', unsafe_allow_html=True)

    fig_roll = go.Figure()
    base = index_cols[0]
    for i, col in enumerate(index_cols[1:], 1):
        roll_corr = returns[base].rolling(252).corr(returns[col])
        fig_roll.add_trace(go.Scatter(
            x=roll_corr.index, y=roll_corr,
            name=col, line=dict(color=COLORS[i % len(COLORS)], width=1.5),
        ))
    if dxy_col:
        roll_dxy = returns[base].rolling(252).corr(returns[dxy_col])
        fig_roll.add_trace(go.Scatter(
            x=roll_dxy.index, y=roll_dxy,
            name="DXY", line=dict(color="#fbbf24", width=1.5, dash="dot"),
        ))
    fig_roll.add_hline(y=0, line_dash="dash", line_color="#6b8cba", line_width=0.8)
    fig_roll.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
        height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45", range=[-1, 1]),
        margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
    )
    st.plotly_chart(fig_roll, use_container_width=True)

    # Rolling correlation (first index vs others)
    st.markdown('<div class="section-header">Скользящая корреляция с DXY (252 дня)</div>', unsafe_allow_html=True)

    fig_roll = go.Figure()
    for i, col in enumerate(index_cols, 1):
        roll_corr = returns[dxy_col].rolling(252).corr(returns[col])
        fig_roll.add_trace(go.Scatter(
            x=roll_corr.index, y=roll_corr,
            name=col, line=dict(color=COLORS[i % len(COLORS)], width=1.5),
        ))
    fig_roll.add_hline(y=0, line_dash="dash", line_color="#6b8cba", line_width=0.8)
    fig_roll.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
        height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45", range=[-1, 1]),
        margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
    )
    st.plotly_chart(fig_roll, use_container_width=True)

    # Insights
    avg_corr_idx = corr_matrix.loc[index_cols, index_cols].values.copy()
    np.fill_diagonal(avg_corr_idx, np.nan)
    mean_corr = np.nanmean(avg_corr_idx)

    if dxy_col:
        dxy_corrs = corr_matrix.loc[index_cols, dxy_col]
        dxy_text = ", ".join([f"{c}: {dxy_corrs[c]:.2f}" for c in index_cols])
    else:
        dxy_text = "—"

    st.markdown(f"""
    <div class="insight-box">
    <strong>📊 Вывод о корреляции:</strong><br>
    Средняя попарная корреляция между выбранными индексами: <strong>{mean_corr:.2f}</strong>.
    {"<br>⚠️ Высокая корреляция (>0.6) снижает эффект диверсификации." if mean_corr > 0.6 else
     "<br>✅ Умеренная корреляция указывает на наличие эффекта диверсификации." if mean_corr > 0.3 else
     "<br>✅ Низкая корреляция — диверсификация должна быть эффективной."}
    <br><br>
    <strong>💵 Корреляция с DXY:</strong> {dxy_text}<br>
    Отрицательная корреляция с долларом означает, что рост доллара сопровождается снижением индексов — важный фактор для портфеля в USD.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">Pair Plot (попарные распределения)</div>', unsafe_allow_html=True)
    fig_pair = px.scatter_matrix(
        returns[corr_cols].dropna() * 100,
        dimensions=corr_cols,
        color_discrete_sequence=["#3b82f6"],
        labels={c: c[:12] for c in corr_cols},
    )
    fig_pair.update_traces(
        diagonal_visible=True,
        showupperhalf=False,
        marker=dict(size=2, opacity=0.3, color="#3b82f6"),
    )
    fig_pair.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
        height=600, margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_pair, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Портфель
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Состав портфеля</div>', unsafe_allow_html=True)

    if weight_method == "Равные веса (1/N)":
        port_ret, weights = build_equal_weight_portfolio(index_returns)
        weight_label = "Равные веса"
    elif weight_method == "Максимальный Sharpe (оптимизация)":
        port_ret, weights = max_sharpe_portfolio(index_returns, rf=rf_rate)
        weight_label = "Оптимальные веса (макс. Sharpe)"
    else:
        port_ret, weights = corr_weight_portfolio(index_returns)
        weight_label = "Веса обратно пропорциональны средней корреляции"

    # Pie chart
    fig_pie = go.Figure(go.Pie(
        labels=index_cols,
        values=weights * 100,
        hole=0.52,
        marker=dict(colors=COLORS[:len(index_cols)], line=dict(color="#0a0e1a", width=2)),
        textinfo="label+percent",
        textfont=dict(size=11, family="IBM Plex Sans"),
    ))
    fig_pie.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        annotations=[dict(text=weight_label[:15]+"…" if len(weight_label)>15 else weight_label,
                          x=0.5, y=0.5, font_size=10, showarrow=False, font_color="#6b8cba")],
        showlegend=False,
    )
    col_pie, col_info = st.columns([1, 1])
    with col_pie:
        st.plotly_chart(fig_pie, use_container_width=True)
    with col_info:
        st.markdown("**Веса портфеля:**")
        for i, (name, w) in enumerate(zip(index_cols, weights)):
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{name}</div>"
                f"<div class='metric-value neutral'>{w*100:.1f}%</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-header">Ручная настройка весов</div>', unsafe_allow_html=True)
    with st.expander("🎚 Задать веса вручную"):
        manual_weights = []
        cols_w = st.columns(len(index_cols))
        for i, col in enumerate(index_cols):
            w_val = st.number_input(col[:20], min_value=0.0, max_value=1.0,
                                    value=float(round(weights[i], 3)),
                                    step=0.05, key=f"w_{col}", format="%.2f")
            manual_weights.append(w_val)
        total_w = sum(manual_weights)
        st.caption(f"Сумма весов: {total_w:.2f} {'✅' if abs(total_w-1)<0.01 else '⚠️ нормализуется автоматически'}")
        if total_w > 0:
            manual_weights = [w / total_w for w in manual_weights]
            port_ret = index_returns.dot(np.array(manual_weights))
            weights = np.array(manual_weights)

    # Portfolio vs indices cumulative
    st.markdown('<div class="section-header">Кумулятивная доходность</div>', unsafe_allow_html=True)

    cum_port = (1 + port_ret).cumprod() * 100
    fig_cum = go.Figure()
    for i, col in enumerate(index_cols):
        cum_idx = (1 + index_returns[col]).cumprod() * 100
        fig_cum.add_trace(go.Scatter(
            x=cum_idx.index, y=cum_idx,
            name=col, line=dict(color=COLORS[i], width=1.2, dash="dot"), opacity=0.6,
        ))
    fig_cum.add_trace(go.Scatter(
        x=cum_port.index, y=cum_port,
        name="🗂 Портфель", line=dict(color="#ffffff", width=2.5),
    ))
    fig_cum.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
        height=360, legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45"),
        margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
    )
    st.plotly_chart(fig_cum, use_container_width=True)

    # Risk-Return scatter
    st.markdown('<div class="section-header">Доходность / Риск (Risk-Return)</div>', unsafe_allow_html=True)

    scatter_data = []
    for col in index_cols:
        scatter_data.append({
            "Индекс": col,
            "Доходность (% год.)": annual_return(index_returns[col]) * 100,
            "Волатильность (% год.)": annual_vol(index_returns[col]) * 100,
            "Sharpe": sharpe(index_returns[col], rf_rate),
        })
    scatter_data.append({
        "Индекс": "🗂 Портфель",
        "Доходность (% год.)": annual_return(port_ret) * 100,
        "Волатильность (% год.)": annual_vol(port_ret) * 100,
        "Sharpe": sharpe(port_ret, rf_rate),
    })
    scatter_df = pd.DataFrame(scatter_data)

    fig_scatter = px.scatter(
        scatter_df, x="Волатильность (% год.)", y="Доходность (% год.)",
        text="Индекс", size_max=14,
        color="Sharpe", color_continuous_scale="RdYlGn",
        hover_data={"Sharpe": ":.2f"},
    )
    fig_scatter.update_traces(textposition="top center", marker=dict(size=12))
    fig_scatter.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
        height=400, xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45"),
        margin=dict(l=0, r=0, t=10, b=0),
        coloraxis_colorbar=dict(title="Sharpe", tickfont=dict(color="#e0e6f0")),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Статистика
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Сводная таблица показателей</div>', unsafe_allow_html=True)

    rows = []
    for col in index_cols:
        r = index_returns[col]
        rows.append({
            "Индекс": col,
            "CAGR (% год.)": f"{annual_return(r)*100:.2f}%",
            "Волатильность (% год.)": f"{annual_vol(r)*100:.2f}%",
            "Коэф. Шарпа": f"{sharpe(r, rf_rate):.2f}",
            "Макс. просадка": f"{max_drawdown(prices[col])*100:.1f}%",
            "Ср. дневная дох.": f"{r.mean()*100:.3f}%",
        })
    # Portfolio row
    r = port_ret
    rows.append({
        "Индекс": "🗂 Портфель",
        "CAGR (% год.)": f"{annual_return(r)*100:.2f}%",
        "Волатильность (% год.)": f"{annual_vol(r)*100:.2f}%",
        "Коэф. Шарпа": f"{sharpe(r, rf_rate):.2f}",
        "Макс. просадка": f"{max_drawdown((1+r).cumprod())*100:.1f}%",
        "Ср. дневная дох.": f"{r.mean()*100:.3f}%",
    })
    stats_df = pd.DataFrame(rows)
    st.dataframe(
        stats_df.set_index("Индекс"),
        use_container_width=True,
        height=38 * (len(rows) + 1) + 10,
    )

    # Drawdown chart
    st.markdown('<div class="section-header">Просадки портфеля и индексов</div>', unsafe_allow_html=True)

    fig_dd = go.Figure()
    for i, col in enumerate(index_cols):
        roll_max = prices[col].cummax()
        dd = (prices[col] - roll_max) / roll_max * 100
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd, name=col,
            line=dict(color=COLORS[i], width=1.2), opacity=0.55,
            fill=None,
        ))
    port_price = (1 + port_ret).cumprod()
    roll_max_p = port_price.cummax()
    dd_p = (port_price - roll_max_p) / roll_max_p * 100
    fig_dd.add_trace(go.Scatter(
        x=dd_p.index, y=dd_p, name="🗂 Портфель",
        line=dict(color="#ffffff", width=2), fill="tozeroy", fillcolor="rgba(255,255,255,0.04)",
    ))
    fig_dd.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
        height=340, legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45", title="Просадка (%)"),
        margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # Summary conclusion
    mean_corr_val = np.nanmean(
        np.where(np.eye(len(index_cols), dtype=bool),
                 np.nan,
                 corr_matrix.loc[index_cols, index_cols].values.copy())
    )
    p_sharpe   = sharpe(port_ret, rf_rate)
    best_idx   = max(index_cols, key=lambda c: sharpe(index_returns[c], rf_rate))
    best_sh    = sharpe(index_returns[best_idx], rf_rate)
    port_vol   = annual_vol(port_ret) * 100
    best_vol   = min(annual_vol(index_returns[c]) for c in index_cols) * 100

    st.markdown('<div class="section-header">Value at Risk (VaR) и CVaR</div>', unsafe_allow_html=True)

    confidence = st.slider("Уровень доверия VaR", 0.90, 0.99, 0.95, 0.01, format="%.2f")
    
    var_rows = []
    for col in index_cols:
        v, cv = var_cvar(index_returns[col], confidence)
        var_rows.append({
            "Индекс": col,
            f"VaR {confidence:.0%} (дневной)": f"{v*100:.2f}%",
            f"CVaR {confidence:.0%} (дневной)": f"{cv*100:.2f}%",
            "VaR (годовой, ×√252)": f"{v*np.sqrt(252)*100:.2f}%",
        })
    pv, pcv = var_cvar(port_ret, confidence)
    var_rows.append({
        "Индекс": "🗂 Портфель",
        f"VaR {confidence:.0%} (дневной)": f"{pv*100:.2f}%",
        f"CVaR {confidence:.0%} (дневной)": f"{pcv*100:.2f}%",
        "VaR (годовой, ×√252)": f"{pv*np.sqrt(252)*100:.2f}%",
    })
    
    var_df = pd.DataFrame(var_rows).set_index("Индекс")
    st.dataframe(var_df, use_container_width=True)
    
    # Распределение доходностей с отметкой VaR
    st.markdown('<div class="section-header">Распределение доходностей портфеля</div>', unsafe_allow_html=True)
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(
        x=port_ret * 100, nbinsx=80,
        marker_color="#3b82f6", opacity=0.7, name="Доходности",
    ))
    fig_hist.add_vline(x=pv*100, line_color="#f87171", line_width=2, line_dash="dash",
                       annotation_text=f"VaR {confidence:.0%}: {pv*100:.2f}%",
                       annotation_font_color="#f87171")
    fig_hist.add_vline(x=pcv*100, line_color="#f59e0b", line_width=1.5, line_dash="dot",
                       annotation_text=f"CVaR: {pcv*100:.2f}%",
                       annotation_font_color="#f59e0b", annotation_position="bottom right")
    fig_hist.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
        height=300, xaxis_title="Дневная доходность (%)",
        xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45"),
        margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown(f"""
    <div class="insight-box">
    <strong>📋 Итоговые выводы</strong><br><br>
    <strong>1. Эффективность международной диверсификации:</strong><br>
    Средняя корреляция между выбранными индексами составляет <strong>{mean_corr_val:.2f}</strong>.
    {"Корреляция умеренная/высокая, что характерно для глобализированных рынков — кризисы бьют по всем индексам одновременно."
     if mean_corr_val > 0.5 else
     "Корреляция относительно низкая — диверсификация по странам даёт реальный эффект снижения риска."}
    <br><br>
    <strong>2. Доходность и риск портфеля:</strong><br>
    Портфель ({weight_label.lower()}) показывает волатильность <strong>{port_vol:.1f}%</strong> годовых при
    коэффициенте Шарпа <strong>{p_sharpe:.2f}</strong>.
    {"Портфель обеспечивает лучший Sharpe, чем лучший отдельный индекс — диверсификация работает." if p_sharpe >= best_sh * 0.9 else
     f"Лучший индекс ({best_idx}) имеет Sharpe {best_sh:.2f} — целевое отдельное вложение было бы выгоднее по Sharpe."}
    Минимальная волатильность среди отдельных индексов: {best_vol:.1f}%, портфель: {port_vol:.1f}%.
    <br><br>
    <strong>3. Влияние DXY:</strong><br>
    {"Индекс доллара имеет, как правило, отрицательную корреляцию с большинством рынков: укрепление USD → отток капитала с EM, давление на сырьевые активы и европейские рынки. Это важно учитывать при формировании валютной экспозиции портфеля." if include_dxy else "DXY не включён в анализ."}
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB EF — Efficient Frontier
# ══════════════════════════════════════════════════════════════════════════════
with tab_ef:
    st.markdown('<div class="section-header">Граница эффективных портфелей</div>',
                unsafe_allow_html=True)

    if len(index_cols) < 2:
        st.warning("Нужно минимум 2 индекса.")
    else:
        with st.spinner("Оптимизация портфелей (3000 симуляций + граница)…"):
            fvols, frets, rvols, rrets, w_sh, w_mv = compute_efficient_frontier(
                index_returns, rf=rf_rate
            )

        mean_ann = index_returns.mean() * 252
        cov_ann  = index_returns.cov()  * 252

        # Sharpe для раскраски случайных портфелей
        rand_sharpes = (rrets - rf_rate) / np.where(rvols > 0, rvols, np.nan)

        fig_ef = go.Figure()

        # Облако случайных портфелей
        fig_ef.add_trace(go.Scatter(
            x=rvols * 100, y=rrets * 100,
            mode="markers",
            marker=dict(
                size=3, opacity=0.45,
                color=rand_sharpes,
                colorscale="RdYlGn",
                cmin=min(rand_sharpes) - 0.1, cmax=max(rand_sharpes) + 0.1,
                # colorbar=dict(
                #     title="Sharpe", thickness=12,
                #     tickfont=dict(color="#e0e6f0"),
                # ),
            ),
            name="Случайные портфели",
            hovertemplate="σ: %{x:.2f}%<br>r: %{y:.2f}%<extra></extra>",
        ))

        # Кривая границы
        fig_ef.add_trace(go.Scatter(
            x=fvols * 100, y=frets * 100,
            mode="lines",
            line=dict(color="#60a5fa", width=2.5),
            name="Efficient Frontier",
            hovertemplate="σ: %{x:.2f}%<br>r: %{y:.2f}%<extra>Frontier</extra>",
        ))

        # Точка: мин. дисперсия
        vol_mv = np.sqrt(w_mv @ cov_ann.values @ w_mv) * 100
        ret_mv = (w_mv @ mean_ann.values) * 100
        fig_ef.add_trace(go.Scatter(
            x=[vol_mv], y=[ret_mv],
            mode="markers+text",
            marker=dict(size=13, color="#34d399", symbol="diamond",
                        line=dict(color="#fff", width=1.5)),
            text=["Min Variance"], textposition="top right",
            textfont=dict(color="#34d399", size=11),
            name="Min Variance",
            hovertemplate=f"Min Variance<br>σ: {vol_mv:.2f}%<br>r: {ret_mv:.2f}%<extra></extra>",
        ))

        # Точка: макс. Sharpe
        vol_sh = np.sqrt(w_sh @ cov_ann.values @ w_sh) * 100
        ret_sh = (w_sh @ mean_ann.values) * 100
        sh_val = (ret_sh/100 - rf_rate) / (vol_sh/100)
        fig_ef.add_trace(go.Scatter(
            x=[vol_sh], y=[ret_sh],
            mode="markers+text",
            marker=dict(size=13, color="#f59e0b", symbol="star",
                        line=dict(color="#fff", width=1.5)),
            text=["Max Sharpe"], textposition="top right",
            textfont=dict(color="#f59e0b", size=11),
            name=f"Max Sharpe ({sh_val:.2f})",
            hovertemplate=f"Max Sharpe<br>σ: {vol_sh:.2f}%<br>r: {ret_sh:.2f}%<br>Sharpe: {sh_val:.2f}<extra></extra>",
        ))

        # Индивидуальные индексы
        for i, col in enumerate(index_cols):
            v = np.sqrt(cov_ann.loc[col, col]) * 100
            r = mean_ann[col] * 100
            fig_ef.add_trace(go.Scatter(
                x=[v], y=[r],
                mode="markers+text",
                marker=dict(size=9, color=COLORS[i % len(COLORS)],
                            symbol="circle", line=dict(color="#fff", width=1)),
                text=[col[:14]], textposition="middle right",
                textfont=dict(color=COLORS[i % len(COLORS)], size=10),
                name=col,
                hovertemplate=f"{col}<br>σ: {v:.2f}%<br>r: {r:.2f}%<extra></extra>",
            ))

        # Capital Market Line
        cml_vols = np.linspace(0, max(rvols) * 1.05, 100) * 100
        cml_rets = rf_rate * 100 + (ret_sh - rf_rate * 100) / vol_sh * cml_vols
        fig_ef.add_trace(go.Scatter(
            x=cml_vols, y=cml_rets,
            mode="lines",
            line=dict(color="#f59e0b", width=1.2, dash="dash"),
            name="CML (Capital Market Line)",
            hovertemplate="CML<br>σ: %{x:.2f}%<br>r: %{y:.2f}%<extra></extra>",
        ))

        fig_ef.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#0a0e1a",
            height=520,
            xaxis=dict(title="Волатильность (% годовых)", gridcolor="#1e2d45"),
            yaxis=dict(title="Доходность (% годовых)",    gridcolor="#1e2d45"),
            legend=dict(orientation="v", x=1.01, y=0.85,
                        font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
            margin=dict(l=0, r=150, t=10, b=0),
            hovermode="closest",
        )
        st.plotly_chart(fig_ef, use_container_width=True)

        # Оптимальные веса
        st.markdown('<div class="section-header">Оптимальные веса</div>',
                    unsafe_allow_html=True)

        col_sh, col_mv = st.columns(2)
        with col_sh:
            st.markdown("**⭐ Макс. Sharpe**")
            fig_sh = go.Figure(go.Bar(
                x=index_cols,
                y=w_sh * 100,
                marker_color=COLORS[:len(index_cols)],
                text=[f"{w*100:.1f}%" for w in w_sh],
                textposition="outside",
            ))
            fig_sh.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
                showlegend=False,
                yaxis=dict(title="%", gridcolor="#1e2d45"),
                xaxis=dict(gridcolor="#1e2d45", tickangle=-25),
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig_sh, use_container_width=True)
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>Доходность / Риск / Sharpe</div>"
                f"<div class='metric-value neutral'>"
                f"{ret_sh:.1f}% / {vol_sh:.1f}% / {sh_val:.2f}"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        with col_mv:
            st.markdown("**💚 Мин. дисперсия**")
            sh_mv = (ret_mv/100 - rf_rate) / (vol_mv/100)
            fig_mv = go.Figure(go.Bar(
                x=index_cols,
                y=w_mv * 100,
                marker_color=COLORS[:len(index_cols)],
                text=[f"{w*100:.1f}%" for w in w_mv],
                textposition="outside",
            ))
            fig_mv.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
                showlegend=False,
                yaxis=dict(title="%", gridcolor="#1e2d45"),
                xaxis=dict(gridcolor="#1e2d45", tickangle=-25),
                margin=dict(l=0, r=0, t=20, b=0),
            )
            st.plotly_chart(fig_mv, use_container_width=True)
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>Доходность / Риск / Sharpe</div>"
                f"<div class='metric-value neutral'>"
                f"{ret_mv:.1f}% / {vol_mv:.1f}% / {sh_mv:.2f}"
                f"</div></div>",
                unsafe_allow_html=True,
            )

        # Вывод
        st.markdown(f"""
        <div class="insight-box">
        <strong>📐 Как читать график:</strong><br>
        Каждая точка — портфель с определёнными весами. <strong>Цвет = коэффициент Sharpe</strong> (зелёный = лучше).
        <strong>Синяя кривая</strong> — граница эффективности: левее и выше неё портфелей не существует.
        Портфели <em>ниже</em> кривой неэффективны — при том же риске можно получить бо́льшую доходность.
        <br><br>
        <strong>⭐ Макс. Sharpe</strong> ({ret_sh:.1f}% / {vol_sh:.1f}% / Sharpe {sh_val:.2f}) —
        точка касания <em>Capital Market Line</em> с фронтиром. Оптимальный портфель для инвестора с безрисковой ставкой {rf_rate*100:.1f}%.<br>
        <strong>💚 Мин. дисперсия</strong> ({ret_mv:.1f}% / {vol_mv:.1f}%) —
        портфель с наименьшим возможным риском. Подходит наиболее консервативным инвесторам.
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Rolling Beta
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Rolling Beta к бенчмарку</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
    <strong>Что такое Beta?</strong><br>
    Beta показывает, насколько индекс движется вместе с рынком-бенчмарком (обычно S&P 500).
    <br>β = 1 → индекс повторяет бенчмарк. β > 1 → амплитуда выше (агрессивнее). β < 1 → менее чувствителен.
    β < 0 → движется против рынка (редкость, ценна для хеджирования).
    <br><br>
    <strong>Почему Rolling?</strong><br>
    Beta не постоянна — она меняется в разные рыночные режимы. Rolling Beta (скользящее окно 252 дня = 1 год)
    показывает, как чувствительность индекса к S&P менялась во времени. В кризис 2020 большинство индексов
    резко повысили бету — это видно на графике.
    <br><br>
    <strong>Формула:</strong> β = Cov(R_i, R_m) / Var(R_m), где R_m — доходность бенчмарка.
    </div>
    """, unsafe_allow_html=True)

    # Load benchmark (S&P 500)
    bench_options = {
        "S&P 500 (^GSPC)": "^GSPC",
        "MSCI World (URTH)": "URTH",
        "Euro Stoxx 50 (^STOXX50E)": "^STOXX50E",
    }
    bench_name = st.selectbox("Бенчмарк", list(bench_options.keys()))
    bench_ticker = bench_options[bench_name]
    roll_window = st.slider("Окно (дней)", 60, 504, 252, 21)

    with st.spinner("Загрузка бенчмарка…"):
        bench_raw = load_data([bench_ticker], start_date, end_date)
        if isinstance(bench_raw.columns, pd.MultiIndex):
            bench_prices = bench_raw["Close"].squeeze()
        else:
            bench_prices = bench_raw.squeeze()
    bench_ret = bench_prices.pct_change().dropna()

    # Align
    common_idx = index_returns.index.intersection(bench_ret.index)
    idx_ret_al  = index_returns.loc[common_idx]
    bench_al    = bench_ret.loc[common_idx]

    # Rolling beta
    fig_beta = go.Figure()
    static_betas = {}
    for i, col in enumerate(index_cols):
        if col == bench_name:
            continue
        roll_cov = idx_ret_al[col].rolling(roll_window).cov(bench_al)
        roll_var = bench_al.rolling(roll_window).var()
        roll_beta = roll_cov / roll_var
        static_betas[col] = (roll_cov.iloc[-1] * roll_window) / (roll_var.iloc[-1] * roll_window) if roll_var.iloc[-1] != 0 else np.nan

        fig_beta.add_trace(go.Scatter(
            x=roll_beta.index, y=roll_beta,
            name=col, line=dict(color=COLORS[i % len(COLORS)], width=1.8),
            hovertemplate=f"{col}<br>Beta: %{{y:.2f}}<extra></extra>",
        ))

    fig_beta.add_hline(y=1.0, line_dash="dash", line_color="#6b8cba", line_width=1,
                       annotation_text="β=1 (повторяет бенчмарк)", annotation_font_color="#6b8cba")
    fig_beta.add_hline(y=0.0, line_dash="dot", line_color="#374151", line_width=0.8)
    fig_beta.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
        height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45", title="Beta (β)"),
        margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
    )
    st.plotly_chart(fig_beta, use_container_width=True)

    # Static beta table + interpretation
    st.markdown('<div class="section-header">Текущая Beta (последнее окно) и интерпретация</div>', unsafe_allow_html=True)
    b_cols = st.columns(len(static_betas))
    for i, (name, beta_val) in enumerate(static_betas.items()):
        with b_cols[i]:
            if np.isnan(beta_val):
                label, color = "н/д", "neutral"
            elif beta_val > 1.2:
                label, color = "агрессивный", "negative"
            elif beta_val > 0.8:
                label, color = "рыночный", "neutral"
            elif beta_val > 0.3:
                label, color = "защитный", "positive"
            else:
                label, color = "некоррелирован", "positive"
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-label'>{name[:22]}</div>"
                f"<div class='metric-value {color}'>{beta_val:.2f}</div>"
                f"<div style='font-size:0.72rem;color:#6b8cba;margin-top:4px'>{label}</div>"
                f"</div>", unsafe_allow_html=True,
            )

    # Scatter: beta vs alpha
    st.markdown('<div class="section-header">Beta vs Alpha (Jensen\'s Alpha)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box" style="margin-bottom:1rem">
    <strong>Jensen's Alpha</strong> — доходность индекса сверх того, что объясняется бетой.
    α = R_i − [R_f + β·(R_m − R_f)]. Положительная альфа = индекс "обыгрывает" рынок с учётом риска.
    </div>
    """, unsafe_allow_html=True)

    scatter_beta = []
    full_bench_ret = bench_al.mean() * 252
    for col in index_cols:
        if len(idx_ret_al[col].dropna()) < 60:
            continue
        cov_ = np.cov(idx_ret_al[col].dropna(), bench_al.loc[idx_ret_al[col].dropna().index])[0, 1]
        var_ = bench_al.var()
        beta_ = cov_ / var_ if var_ > 0 else np.nan
        ann_r = annual_return(idx_ret_al[col])
        alpha_ = ann_r - (rf_rate + beta_ * (full_bench_ret - rf_rate))
        scatter_beta.append({"Индекс": col, "Beta": beta_, "Alpha (%)": alpha_ * 100,
                              "Доходность (%)": ann_r * 100})

    sdf = pd.DataFrame(scatter_beta)
    fig_ba = px.scatter(sdf, x="Beta", y="Alpha (%)", text="Индекс",
                        color="Доходность (%)", color_continuous_scale="RdYlGn",
                        size_max=14)
    fig_ba.update_traces(textposition="top center", marker=dict(size=12))
    fig_ba.add_vline(x=1, line_dash="dash", line_color="#6b8cba", line_width=0.8)
    fig_ba.add_hline(y=0, line_dash="dash", line_color="#6b8cba", line_width=0.8)
    fig_ba.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
        height=380, xaxis=dict(gridcolor="#1e2d45", title="Beta (β)"),
        yaxis=dict(gridcolor="#1e2d45", title="Jensen's Alpha (% год.)"),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_ba, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Стохастические модели
# ══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("## Стохастические модели прогнозирования")
    st.markdown("""
    <div class="insight-box">
    Стохастические модели описывают цену актива как случайный процесс с математически заданной структурой.
    Они позволяют <strong>симулировать будущие сценарии</strong>, оценивать риск и строить прогнозы.
    В отличие от технического анализа, эти модели имеют строгую математическую основу и используются
    в банках, хедж-фондах и управляющих компаниях.
    </div>
    """, unsafe_allow_html=True)

    target_col = st.selectbox("Индекс для моделирования", index_cols, key="stoch_target")
    target_ret = index_returns[target_col].dropna()
    target_price = prices[target_col].dropna()

    model_tabs = st.tabs(["📊 GBM + Монте-Карло", "📉 GARCH(1,1)", "🌀 Heston Model"])

    # ── GBM ────────────────────────────────────────────────────────────────────
    with model_tabs[0]:
        st.markdown('<div class="section-header">Geometric Brownian Motion</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
        <strong>Модель:</strong> dS = μ·S·dt + σ·S·dW<br>
        Цена движется с постоянным дрейфом μ (средняя доходность) и случайными колебаниями σ·dW,
        где dW — приращение стандартного броуновского движения (белый шум).<br><br>
        <strong>Дискретная форма:</strong> S(t+1) = S(t) · exp[(μ − σ²/2)·Δt + σ·√Δt·Z], где Z ~ N(0,1)<br><br>
        <strong>Применение:</strong> базовая модель для оценки опционов (Black-Scholes), расчёта VaR,
        стресс-тестирования портфелей. Допущение — постоянная волатильность (GARCH и Heston снимают это ограничение).
        </div>
        """, unsafe_allow_html=True)

        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            n_sim    = st.slider("Число симуляций", 100, 5000, 1000, 100, key="gbm_nsim")
        with col_g2:
            horizon  = st.slider("Горизонт (дней)", 21, 504, 252, 21, key="gbm_hor")
        with col_g3:
            gbm_mode = st.radio("Параметры", ["Из данных", "Вручную"], key="gbm_mode")

        mu_hist  = target_ret.mean() * 252
        sig_hist = target_ret.std()  * np.sqrt(252)

        if gbm_mode == "Вручную":
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                mu_val  = st.number_input("μ (годовая доходность)", value=float(round(mu_hist, 3)),
                                          step=0.01, format="%.3f", key="gbm_mu")
            with col_m2:
                sig_val = st.number_input("σ (годовая волатильность)", value=float(round(sig_hist, 3)),
                                          min_value=0.001, step=0.01, format="%.3f", key="gbm_sig")
        else:
            mu_val, sig_val = mu_hist, sig_hist

        dt = 1 / 252
        S0 = float(target_price.iloc[-1])
        np.random.seed(42)
        Z  = np.random.standard_normal((horizon, n_sim))
        log_ret = (mu_val - 0.5 * sig_val**2) * dt + sig_val * np.sqrt(dt) * Z
        S = S0 * np.exp(np.cumsum(log_ret, axis=0))
        S = np.vstack([np.full(n_sim, S0), S])

        # Percentiles
        p5, p25, p50, p75, p95 = [np.percentile(S, p, axis=1) for p in [5, 25, 50, 75, 95]]
        t_idx = pd.date_range(target_price.index[-1], periods=horizon + 1, freq="B")

        fig_gbm = go.Figure()
        # Fan of random paths (thin, transparent)
        show_paths = min(n_sim, 200)
        for j in range(show_paths):
            fig_gbm.add_trace(go.Scatter(
                x=t_idx, y=S[:, j],
                mode="lines", line=dict(color="#3b82f6", width=0.3),
                opacity=0.08, showlegend=False, hoverinfo="skip",
            ))
        # Confidence bands
        fig_gbm.add_trace(go.Scatter(x=list(t_idx)+list(t_idx[::-1]),
            y=list(p95)+list(p5[::-1]),
            fill="toself", fillcolor="rgba(59,130,246,0.08)",
            line=dict(color="rgba(0,0,0,0)"), name="90% CI", hoverinfo="skip"))
        fig_gbm.add_trace(go.Scatter(x=list(t_idx)+list(t_idx[::-1]),
            y=list(p75)+list(p25[::-1]),
            fill="toself", fillcolor="rgba(59,130,246,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="50% CI", hoverinfo="skip"))
        fig_gbm.add_trace(go.Scatter(x=t_idx, y=p50, name="Медиана",
            line=dict(color="#60a5fa", width=2.5)))
        fig_gbm.add_trace(go.Scatter(x=t_idx, y=p5,  name="5-й перцентиль",
            line=dict(color="#f87171", width=1.5, dash="dash")))
        fig_gbm.add_trace(go.Scatter(x=t_idx, y=p95, name="95-й перцентиль",
            line=dict(color="#34d399", width=1.5, dash="dash")))

        # Historical
        hist_tail = target_price.iloc[-252:]
        fig_gbm.add_trace(go.Scatter(x=hist_tail.index, y=hist_tail,
            name="История (1 год)", line=dict(color="#f59e0b", width=2)))

        fig_gbm.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
            height=440, legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45", title="Цена"),
            margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
        )
        st.plotly_chart(fig_gbm, use_container_width=True)

        # Terminal distribution
        S_T = S[-1, :]
        gbm_var = np.percentile(S_T, 5)
        fig_hist_gbm = go.Figure()
        fig_hist_gbm.add_trace(go.Histogram(x=S_T, nbinsx=80,
            marker_color="#3b82f6", opacity=0.7, name="Конечная цена"))
        fig_hist_gbm.add_vline(x=S0, line_color="#f59e0b", line_dash="dash",
            annotation_text=f"S₀={S0:.1f}", annotation_font_color="#f59e0b")
        fig_hist_gbm.add_vline(x=gbm_var, line_color="#f87171", line_dash="dash",
            annotation_text=f"VaR 95%: {gbm_var:.1f}", annotation_font_color="#f87171")
        fig_hist_gbm.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
            height=280, xaxis_title="Цена через N дней",
            xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45"),
            margin=dict(l=0, r=0, t=20, b=0), title="Распределение конечной цены",
        )
        st.plotly_chart(fig_hist_gbm, use_container_width=True)

        loss_prob = np.mean(S_T < S0) * 100
        st.markdown(f"""
        <div class="insight-box">
        <strong>Параметры:</strong> μ = {mu_val*100:.2f}% годовых, σ = {sig_val*100:.2f}% годовых<br>
        <strong>Медианная цена через {horizon} дней:</strong> {np.median(S_T):.1f} (сейчас: {S0:.1f})<br>
        <strong>Вероятность убытка:</strong> {loss_prob:.1f}%<br>
        <strong>95% VaR (GBM):</strong> цена ниже {gbm_var:.1f} с вероятностью 5%
        </div>
        """, unsafe_allow_html=True)

    # ── GARCH ──────────────────────────────────────────────────────────────────
    with model_tabs[1]:
        st.markdown('<div class="section-header">GARCH(1,1) — Generalized Autoregressive Conditional Heteroskedasticity</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
        <strong>Ключевая идея:</strong> волатильность не постоянна — она <em>кластеризуется</em>.
        После периодов высокой волатильности (кризис) следуют ещё высоковолатильные дни.
        GBM этого не учитывает. GARCH — учитывает.<br><br>
        <strong>Модель GARCH(1,1):</strong><br>
        σ²(t) = ω + α·ε²(t−1) + β·σ²(t−1)<br>
        • ω — базовый уровень волатильности<br>
        • α — реакция на последний шок (ε² — квадрат неожиданной доходности)<br>
        • β — "память" о прошлой волатильности<br>
        • α + β < 1 — условие стационарности; чем ближе к 1, тем дольше шок рассасывается<br><br>
        <strong>Применение:</strong> VaR с учётом текущего режима волатильности, ценообразование деривативов,
        управление риском в реальном времени. Используется Bloomberg, JPMorgan, Barclays.
        </div>
        """, unsafe_allow_html=True)

        if not ARCH_AVAILABLE:
            st.warning("Установи библиотеку: `pip install arch`")
        else:
            garch_horizon = st.slider("Горизонт прогноза волатильности (дней)", 5, 126, 30, 5)

            ret_pct = target_ret * 100  # arch принимает доходности в %

            with st.spinner("Обучение GARCH(1,1)…"):
                am = arch_model(ret_pct, vol="Garch", p=1, q=1, dist="Normal", rescale=False)
                res = am.fit(disp="off")

            omega = res.params.get("omega", res.params.iloc[1])
            alpha = res.params.get("alpha[1]", res.params.iloc[2])
            beta  = res.params.get("beta[1]",  res.params.iloc[3])
            persistence = alpha + beta
            long_run_vol = np.sqrt(omega / (1 - persistence)) if persistence < 1 else np.nan

            # Conditional volatility
            cond_vol = res.conditional_volatility  # % per day

            fig_garch = make_subplots(rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.4, 0.6], vertical_spacing=0.04,
                subplot_titles=["Доходности (%)", "Условная волатильность GARCH(1,1) (% в день)"])

            fig_garch.add_trace(go.Scatter(x=ret_pct.index, y=ret_pct,
                line=dict(color="#60a5fa", width=0.8), name="Доходность"), row=1, col=1)
            fig_garch.add_trace(go.Scatter(x=cond_vol.index, y=cond_vol,
                line=dict(color="#f59e0b", width=1.5), name="σ GARCH",
                fill="tozeroy", fillcolor="rgba(245,158,11,0.1)"), row=2, col=1)
            if not np.isnan(long_run_vol):
                fig_garch.add_hline(y=long_run_vol, line_dash="dash", line_color="#34d399",
                    annotation_text=f"Долгосрочная σ: {long_run_vol:.2f}%", row=2, col=1,
                    annotation_font_color="#34d399")

            fig_garch.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
                height=480, showlegend=False,
                xaxis2=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45"),
                yaxis2=dict(gridcolor="#1e2d45"),
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig_garch, use_container_width=True)

            # Forecast
            forecast = res.forecast(horizon=garch_horizon, reindex=False)
            fc_var = forecast.variance.iloc[-1].values
            fc_vol_daily = np.sqrt(fc_var)
            fc_vol_ann   = fc_vol_daily * np.sqrt(252)
            fc_days = np.arange(1, garch_horizon + 1)

            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=fc_days, y=fc_vol_daily,
                name="Прогноз σ (% в день)", line=dict(color="#f59e0b", width=2),
                mode="lines+markers", marker=dict(size=5)))
            if not np.isnan(long_run_vol):
                fig_fc.add_hline(y=long_run_vol, line_dash="dash", line_color="#34d399",
                    annotation_text="Долгосрочный уровень", annotation_font_color="#34d399")
            current_vol_daily = float(cond_vol.iloc[-1])
            fig_fc.add_hline(y=current_vol_daily, line_dash="dot", line_color="#60a5fa",
                annotation_text=f"Текущая σ: {current_vol_daily:.2f}%", annotation_font_color="#60a5fa")
            fig_fc.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
                height=300, xaxis_title="Дней вперёд", yaxis_title="Волатильность (% в день)",
                xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45"),
                margin=dict(l=0, r=0, t=20, b=0),
                title=f"Прогноз волатильности на {garch_horizon} дней"
            )
            st.plotly_chart(fig_fc, use_container_width=True)

            # Parameters table
            st.markdown('<div class="section-header">Оценённые параметры GARCH(1,1)</div>', unsafe_allow_html=True)
            pc1, pc2, pc3, pc4 = st.columns(4)
            for col_widget, label, val, hint in [
                (pc1, "ω (базовая дисп.)", f"{omega:.4f}", "константа"),
                (pc2, "α (ARCH-эффект)", f"{alpha:.4f}", "шок вчера"),
                (pc3, "β (GARCH-память)", f"{beta:.4f}", "σ вчера"),
                (pc4, "α+β (персист.)", f"{persistence:.4f}",
                 "→1 = долгая память" if persistence > 0.95 else "умеренная память"),
            ]:
                with col_widget:
                    st.markdown(
                        f"<div class='metric-card'><div class='metric-label'>{label}</div>"
                        f"<div class='metric-value neutral'>{val}</div>"
                        f"<div style='font-size:0.72rem;color:#6b8cba;margin-top:4px'>{hint}</div>"
                        f"</div>", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="insight-box" style="margin-top:1rem">
            <strong>Интерпретация:</strong><br>
            Персистентность α+β = <strong>{persistence:.3f}</strong> —
            {"шоки волатильности рассасываются <strong>медленно</strong> (типично для акций в кризис)." if persistence > 0.95
             else "волатильность возвращается к норме относительно быстро."}<br>
            Текущая дневная σ: <strong>{current_vol_daily:.2f}%</strong> (годовая: <strong>{current_vol_daily*np.sqrt(252):.1f}%</strong>).
            {f"Долгосрочная равновесная σ: <strong>{long_run_vol:.2f}%</strong>/день." if not np.isnan(long_run_vol) else ""}
            </div>
            """, unsafe_allow_html=True)

    # ── HESTON ─────────────────────────────────────────────────────────────────
    with model_tabs[2]:
        st.markdown('<div class="section-header">Heston Model — стохастическая волатильность</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
        <strong>Проблема GBM:</strong> волатильность σ постоянна — нереалистично.<br>
        <strong>Решение Heston (1993):</strong> волатильность сама случайна и следует своему процессу.<br><br>
        <strong>Система уравнений:</strong><br>
        dS = μ·S·dt + √v·S·dW₁ &nbsp;← цена (как GBM, но σ=√v теперь случайна)<br>
        dv = κ·(θ−v)·dt + ξ·√v·dW₂ &nbsp;← волатильность (процесс CIR)<br><br>
        <strong>Параметры:</strong><br>
        • κ — скорость возврата к среднему (mean reversion speed)<br>
        • θ — долгосрочный уровень волатильности (long-run variance)<br>
        • ξ (vol of vol) — волатильность самой волатильности<br>
        • ρ — корреляция между dW₁ и dW₂ (обычно отрицательная: рост цены → падение волатильности)<br><br>
        <strong>Почему важно:</strong> модель воспроизводит <em>volatility smile/skew</em> — опционные трейдеры
        видят, что implied vol зависит от страйка. GBM этого дать не может. Heston — стандарт в equity derivatives.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Параметры модели**")
        hcol1, hcol2 = st.columns(2)
        with hcol1:
            h_kappa = st.slider("κ (скорость mean reversion)", 0.1, 10.0, 2.0, 0.1)
            h_theta = st.slider("θ (долгосрочная дисперсия)", 0.01, 0.25,
                                float(round(target_ret.var() * 252, 3)), 0.005)
            h_xi    = st.slider("ξ (vol of vol)", 0.05, 1.5, 0.3, 0.05)
        with hcol2:
            h_rho   = st.slider("ρ (корреляция цена-волатильность)", -0.99, 0.0, -0.7, 0.01)
            h_nsim  = st.slider("Симуляций", 200, 3000, 500, 100, key="heston_nsim")
            h_hor   = st.slider("Горизонт (дней)", 21, 504, 252, 21, key="heston_hor")

        h_mu = float(target_ret.mean() * 252)
        h_v0 = float(target_ret.iloc[-60:].var() * 252)  # current variance
        h_S0 = float(target_price.iloc[-1])
        h_dt = 1 / 252

        # Euler-Maruyama discretisation of Heston
        with st.spinner("Симуляция Heston (Euler-Maruyama)…"):
            np.random.seed(7)
            rng = np.random.default_rng(7)
            Z1  = rng.standard_normal((h_hor, h_nsim))
            Z2_ = rng.standard_normal((h_hor, h_nsim))
            Z2  = h_rho * Z1 + np.sqrt(1 - h_rho**2) * Z2_

            S_h = np.zeros((h_hor + 1, h_nsim))
            v_h = np.zeros((h_hor + 1, h_nsim))
            S_h[0] = h_S0
            v_h[0] = max(h_v0, 1e-6)

            for t in range(h_hor):
                v_pos = np.maximum(v_h[t], 0)
                sv    = np.sqrt(v_pos)
                v_h[t+1] = v_h[t] + h_kappa*(h_theta - v_h[t])*h_dt + h_xi*sv*np.sqrt(h_dt)*Z2[t]
                v_h[t+1] = np.maximum(v_h[t+1], 0)  # reflection
                S_h[t+1] = S_h[t] * np.exp((h_mu - 0.5*v_pos)*h_dt + sv*np.sqrt(h_dt)*Z1[t])

        t_h_idx = pd.date_range(target_price.index[-1], periods=h_hor + 1, freq="B")
        hp5, hp25, hp50, hp75, hp95 = [np.percentile(S_h, p, axis=1) for p in [5, 25, 50, 75, 95]]
        vp5, vp50, vp95 = [np.percentile(np.sqrt(np.maximum(v_h, 0)) * 100, p, axis=1) for p in [5, 50, 95]]

        fig_heston = make_subplots(rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.6, 0.4], vertical_spacing=0.04,
            subplot_titles=["Цена (Heston)", "Реализованная волатильность √v (% годовых)"])

        # Price paths
        show_h = min(h_nsim, 150)
        for j in range(show_h):
            fig_heston.add_trace(go.Scatter(x=t_h_idx, y=S_h[:, j],
                mode="lines", line=dict(color="#a78bfa", width=0.3),
                opacity=0.07, showlegend=False, hoverinfo="skip"), row=1, col=1)
        fig_heston.add_trace(go.Scatter(x=list(t_h_idx)+list(t_h_idx[::-1]),
            y=list(hp95)+list(hp5[::-1]), fill="toself",
            fillcolor="rgba(167,139,250,0.1)", line=dict(color="rgba(0,0,0,0)"),
            name="90% CI"), row=1, col=1)
        fig_heston.add_trace(go.Scatter(x=t_h_idx, y=hp50, name="Медиана (Heston)",
            line=dict(color="#a78bfa", width=2.5)), row=1, col=1)
        fig_heston.add_trace(go.Scatter(x=t_h_idx, y=hp5, name="5-й перцентиль",
            line=dict(color="#f87171", width=1.5, dash="dash")), row=1, col=1)
        hist_t = target_price.iloc[-252:]
        fig_heston.add_trace(go.Scatter(x=hist_t.index, y=hist_t, name="История",
            line=dict(color="#f59e0b", width=2)), row=1, col=1)

        # Vol paths
        fig_heston.add_trace(go.Scatter(x=list(t_h_idx)+list(t_h_idx[::-1]),
            y=list(vp95)+list(vp5[::-1]), fill="toself",
            fillcolor="rgba(52,211,153,0.1)", line=dict(color="rgba(0,0,0,0)"),
            name="90% CI σ", showlegend=False), row=2, col=1)
        fig_heston.add_trace(go.Scatter(x=t_h_idx, y=vp50, name="Медиана σ",
            line=dict(color="#34d399", width=1.8)), row=2, col=1)
        fig_heston.add_hline(y=np.sqrt(h_theta)*100, line_dash="dash", line_color="#f59e0b",
            annotation_text=f"θ (долгосрочная): {np.sqrt(h_theta)*100:.1f}%",
            annotation_font_color="#f59e0b", row=2, col=1)

        fig_heston.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
            height=560, legend=dict(orientation="h", yanchor="bottom", y=1.02),
            xaxis2=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45"),
            yaxis2=dict(gridcolor="#1e2d45", title="σ (% год.)"),
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_heston, use_container_width=True)

        # Compare GBM vs Heston terminal distributions
        st.markdown('<div class="section-header">Сравнение: GBM vs Heston (конечное распределение)</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box">
        Ключевое различие: Heston генерирует более тяжёлые хвосты (fat tails) и асимметрию.
        Это соответствует реальным рынкам, где экстремальные движения случаются чаще, чем предсказывает нормальное распределение.
        </div>
        """, unsafe_allow_html=True)

        gbm_T = S0 * np.exp((mu_val - 0.5*sig_val**2)*h_hor*h_dt
                            + sig_val*np.sqrt(h_hor*h_dt)*np.random.standard_normal(h_nsim))
        heston_T = S_h[-1, :]

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Histogram(x=gbm_T, nbinsx=60, name="GBM",
            marker_color="#3b82f6", opacity=0.55, histnorm="probability density"))
        fig_comp.add_trace(go.Histogram(x=heston_T, nbinsx=60, name="Heston",
            marker_color="#a78bfa", opacity=0.55, histnorm="probability density"))
        fig_comp.add_vline(x=h_S0, line_color="#f59e0b", line_dash="dash",
            annotation_text=f"S₀={h_S0:.0f}", annotation_font_color="#f59e0b")
        fig_comp.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
            height=320, xaxis_title="Конечная цена", barmode="overlay",
            xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45"),
            margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.02),
        )
        st.plotly_chart(fig_comp, use_container_width=True)

        h_loss_prob = np.mean(heston_T < h_S0) * 100
        h_var95     = np.percentile(heston_T, 5)
        gbm_var95   = np.percentile(gbm_T, 5)
        st.markdown(f"""
        <div class="insight-box">
        <strong>Параметры Heston:</strong> κ={h_kappa}, θ={h_theta:.3f}, ξ={h_xi}, ρ={h_rho}<br>
        <strong>Вероятность убытка (Heston):</strong> {h_loss_prob:.1f}%<br>
        <strong>95% VaR — GBM:</strong> {gbm_var95:.1f} &nbsp;|&nbsp;
        <strong>95% VaR — Heston:</strong> {h_var95:.1f}
        {"&nbsp; ← Heston даёт более консервативную оценку риска" if h_var95 < gbm_var95 else ""}<br>
        <strong>ρ = {h_rho}</strong> — {"отрицательная корреляция: рост цены сопровождается падением волатильности (leverage effect, типично для акций)." if h_rho < 0 else "нейтральная корреляция."}
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — Оптимальный портфель (перебор комбинаций)
# ══════════════════════════════════════════════════════════════════════════════
with tab7:
    st.markdown("## 🏆 Подбор оптимального портфеля")
    st.markdown("""
    <div class="insight-box">
    Перебираются <strong>все комбинации</strong> из доступных индексов (2, 3 или 4 актива).
    Для каждой комбинации оптимизируются веса на <strong>максимальный коэффициент Шарпа</strong>
    методом SLSQP с множественными случайными стартами (чтобы не застрять в локальном минимуме).<br><br>
    После нахождения лучшего набора — можно вручную подстроить веса и пересчитать результат.
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        max_assets = st.radio("Макс. активов в портфеле", [2, 3, 4], index=2, horizontal=True)
    with col_s2:
        n_starts = st.slider("Случайных стартов оптимизации", 50, 500, 200, 50,
                             help="Больше = точнее, но медленнее")
    with col_s3:
        available_pool = st.multiselect(
            "Пул индексов для подбора",
            options=universe_cols,
            default=universe_cols,
            help="Исключи индексы, которые не хочешь видеть в портфеле"
        )

    if len(available_pool) < 2:
        st.warning("Выбери минимум 2 индекса в пуле.")
        st.stop()

    n_combos = sum(
        len(list(combinations(available_pool, k)))
        for k in range(2, max_assets + 1)
    )
    # st.caption(f"Будет проверено комбинаций: **{n_combos}** × {n_starts} стартов")

    if st.button("🚀 Найти оптимальный портфель", type="primary"):
        with st.spinner(f"Оптимизация {n_combos} комбинаций…"):
            np.random.seed(42)
            pool_returns = universe_returns[available_pool]
            top_results  = find_best_portfolio(
                pool_returns, rf=rf_rate,
                max_assets=max_assets, n_opt=n_starts
            )
        st.session_state["top_results"] = top_results
        st.session_state["opt_done"]    = True

    if st.session_state.get("opt_done"):
        top_results = st.session_state["top_results"]
        best        = top_results[0]

        # ── Топ-10 таблица ────────────────────────────────────────────────
        st.markdown('<div class="section-header">Топ-10 портфелей по Sharpe</div>',
                    unsafe_allow_html=True)

        table_rows = []
        for i, r in enumerate(top_results):
            w_str = " / ".join([f"{a[:10]}:{w*100:.0f}%" for a, w in zip(r["assets"], r["weights"])])
            table_rows.append({
                "#":          i + 1,
                "Активы и веса": w_str,
                "Sharpe":     f"{r['sharpe']:.3f}",
                "CAGR":       f"{r['cagr']:.2f}%",
                "Волат.":     f"{r['vol']:.2f}%",
                "Макс. просадка": f"{r['max_dd']:.1f}%",
            })
        st.dataframe(pd.DataFrame(table_rows).set_index("#"),
                     use_container_width=True, height=400)

        # ── Детали лучшего ────────────────────────────────────────────────
        st.markdown('<div class="section-header">Лучший портфель — детали</div>',
                    unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        for col_w, label, val, cls in [
            (m1, "Sharpe Ratio",  f"{best['sharpe']:.3f}", "positive"),
            (m2, "CAGR",         f"{best['cagr']:.2f}%",  "positive" if best['cagr'] > 0 else "negative"),
            (m3, "Волатильность",f"{best['vol']:.2f}%",   "neutral"),
            (m4, "Макс. просадка",f"{best['max_dd']:.1f}%","negative"),
        ]:
            with col_w:
                st.markdown(
                    f"<div class='metric-card'><div class='metric-label'>{label}</div>"
                    f"<div class='metric-value {cls}'>{val}</div></div>",
                    unsafe_allow_html=True)

        # Pie chart лучшего
        fig_best_pie = go.Figure(go.Pie(
            labels=best["assets"],
            values=best["weights"] * 100,
            hole=0.5,
            marker=dict(colors=COLORS[:len(best["assets"])],
                        line=dict(color="#0a0e1a", width=2)),
            textinfo="label+percent",
        ))
        fig_best_pie.update_layout(
            template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
            height=300, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
        )

        col_pie2, col_cum2 = st.columns([1, 2])
        with col_pie2:
            st.plotly_chart(fig_best_pie, use_container_width=True)
        with col_cum2:
            cum_best = (1 + best["returns"]).cumprod() * 100
            fig_cum_best = go.Figure()
            for i, a in enumerate(best["assets"]):
                cum_a = (1 + universe_returns[a]).cumprod() * 100
                fig_cum_best.add_trace(go.Scatter(x=cum_a.index, y=cum_a,
                    name=a, line=dict(color=COLORS[i], width=1.2, dash="dot"), opacity=0.6))
            fig_cum_best.add_trace(go.Scatter(x=cum_best.index, y=cum_best,
                name="Оптимальный портфель",
                line=dict(color="#ffffff", width=2.5)))
            fig_cum_best.update_layout(
                template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
                height=300, legend=dict(orientation="h", yanchor="bottom", y=1.02),
                xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45"),
                margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
            )
            st.plotly_chart(fig_cum_best, use_container_width=True)

        # ── Ручная настройка весов ─────────────────────────────────────────
        st.markdown('<div class="section-header">🎚 Ручная настройка весов</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="insight-box" style="margin-bottom:1rem">
        Выбери любой набор индексов и задай веса вручную — результат пересчитается мгновенно.
        Можно использовать как стартовую точку оптимальный портфель или задать свою комбинацию.
        </div>
        """, unsafe_allow_html=True)

        manual_assets = st.multiselect(
            "Активы для ручного портфеля",
            options=universe_cols,
            default=best["assets"],
            max_selections=8,
            key="manual_assets_selector"
        )

        if manual_assets:
            st.markdown("**Задай веса (автоматически нормализуются до 100%)**")
            manual_w = []
            w_cols   = st.columns(len(manual_assets))
            # Pre-fill: if same assets as best, use optimal weights; else equal
            prefill = {}
            if set(manual_assets) == set(best["assets"]):
                for a, w in zip(best["assets"], best["weights"]):
                    prefill[a] = float(round(w * 100, 1))

            for i, asset in enumerate(manual_assets):
                default_w = prefill.get(asset, round(100 / len(manual_assets), 1))
                with w_cols[i]:
                    w_val = st.number_input(
                        asset[:18], min_value=0.0, max_value=100.0,
                        value=default_w, step=5.0, format="%.1f",
                        key=f"manual_w_{asset}"
                    )
                    manual_w.append(w_val)

            total_w = sum(manual_w)
            if total_w > 0:
                norm_w = np.array(manual_w) / total_w
                manual_ret = universe_returns[manual_assets].dropna().dot(norm_w)

                # Live metrics
                man_sh  = sharpe(manual_ret, rf_rate)
                man_cagr = annual_return(manual_ret) * 100
                man_vol  = annual_vol(manual_ret) * 100
                man_dd   = max_drawdown((1 + manual_ret).cumprod()) * 100

                st.caption(f"Сумма: {total_w:.1f}% → нормализовано до 100%")
                mm1, mm2, mm3, mm4 = st.columns(4)
                for col_w2, label, val, cls in [
                    (mm1, "Sharpe",        f"{man_sh:.3f}",   "positive" if man_sh > 0 else "negative"),
                    (mm2, "CAGR",          f"{man_cagr:.2f}%","positive" if man_cagr > 0 else "negative"),
                    (mm3, "Волатильность", f"{man_vol:.2f}%", "neutral"),
                    (mm4, "Макс. просадка",f"{man_dd:.1f}%",  "negative"),
                ]:
                    with col_w2:
                        st.markdown(
                            f"<div class='metric-card'><div class='metric-label'>{label}</div>"
                            f"<div class='metric-value {cls}'>{val}</div>"
                            f"<div style='font-size:0.72rem;color:#6b8cba;margin-top:4px'>"
                            f"{'▲ vs оптимальный' if label == 'Sharpe' and man_sh > best['sharpe'] else ''}"
                            f"</div></div>",
                            unsafe_allow_html=True)

                # Cumulative chart manual vs best
                cum_man  = (1 + manual_ret).cumprod() * 100
                cum_opt  = (1 + best["returns"]).cumprod() * 100
                fig_man  = go.Figure()
                for i, asset in enumerate(manual_assets):
                    ca = (1 + universe_returns[asset]).cumprod() * 100
                    fig_man.add_trace(go.Scatter(x=ca.index, y=ca, name=asset,
                        line=dict(color=COLORS[i], width=1, dash="dot"), opacity=0.5))
                fig_man.add_trace(go.Scatter(x=cum_opt.index, y=cum_opt,
                    name=f"Оптимальный (Sharpe {best['sharpe']:.2f})",
                    line=dict(color="#f59e0b", width=2, dash="dash")))
                fig_man.add_trace(go.Scatter(x=cum_man.index, y=cum_man,
                    name=f"Ручной (Sharpe {man_sh:.2f})",
                    line=dict(color="#ffffff", width=2.5)))
                fig_man.update_layout(
                    template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#0a0e1a",
                    height=360, legend=dict(orientation="h", yanchor="bottom", y=1.02),
                    xaxis=dict(gridcolor="#1e2d45"), yaxis=dict(gridcolor="#1e2d45"),
                    margin=dict(l=0, r=0, t=10, b=0), hovermode="x unified",
                )
                st.plotly_chart(fig_man, use_container_width=True)
