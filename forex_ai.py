"""
Robot analisa forex (gratis): grafik real-time TradingView + robot sinyal.

Instal:   py -m pip install streamlit yfinance pandas plotly tradingview-ta
Jalankan: py -m streamlit run forex_ai.py
"""
import json
import math
import urllib.request

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
from plotly.subplots import make_subplots

st.set_page_config(page_title="Robot analisa forex", layout="wide")

PAIRS = {
    "EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X",
    "AUD/USD": "AUDUSD=X", "USD/CAD": "USDCAD=X", "USD/CHF": "USDCHF=X",
    "NZD/USD": "NZDUSD=X", "EUR/JPY": "EURJPY=X", "GBP/JPY": "GBPJPY=X",
    "XAU/USD (emas)": "GC=F",
}
# label: (interval yfinance, periode, resample)
TIMEFRAMES = {
    "15 menit": ("15m", "30d", None),
    "1 jam": ("1h", "120d", None),
    "4 jam": ("1h", "360d", "4h"),
    "Harian": ("1d", "2y", None),
    "Mingguan": ("1wk", "5y", None),
}


# pair: (simbol widget, simbol TA, screener, exchange)
TV = {
    "EUR/USD": ("FX:EURUSD", "EURUSD", "forex", "FX_IDC"),
    "GBP/USD": ("FX:GBPUSD", "GBPUSD", "forex", "FX_IDC"),
    "USD/JPY": ("FX:USDJPY", "USDJPY", "forex", "FX_IDC"),
    "AUD/USD": ("FX:AUDUSD", "AUDUSD", "forex", "FX_IDC"),
    "USD/CAD": ("FX:USDCAD", "USDCAD", "forex", "FX_IDC"),
    "USD/CHF": ("FX:USDCHF", "USDCHF", "forex", "FX_IDC"),
    "NZD/USD": ("FX:NZDUSD", "NZDUSD", "forex", "FX_IDC"),
    "EUR/JPY": ("FX:EURJPY", "EURJPY", "forex", "FX_IDC"),
    "GBP/JPY": ("FX:GBPJPY", "GBPJPY", "forex", "FX_IDC"),
    "XAU/USD (emas)": ("OANDA:XAUUSD", "XAUUSD", "cfd", "OANDA"),
}
# label: (interval widget, nama konstanta Interval tradingview-ta)
TV_TF = {
    "15 menit": ("15", "INTERVAL_15_MINUTES"),
    "1 jam": ("60", "INTERVAL_1_HOUR"),
    "4 jam": ("240", "INTERVAL_4_HOURS"),
    "Harian": ("D", "INTERVAL_1_DAY"),
    "Mingguan": ("W", "INTERVAL_1_WEEK"),
}


def tv_chart_html(symbol, interval, dark):
    cfg = {
        "autosize": True, "symbol": symbol, "interval": interval,
        "timezone": "Asia/Jakarta", "theme": "dark" if dark else "light",
        "style": "1", "locale": "id", "allow_symbol_change": False,
        "withdateranges": True, "support_host": "https://www.tradingview.com",
        "studies": ["MASimple@tv-basicstudies", "RSI@tv-basicstudies", "MACD@tv-basicstudies"],
    }
    return (
        '<div style="height:620px"><div class="tradingview-widget-container" '
        'style="height:100%;width:100%"><div class="tradingview-widget-container__widget" '
        'style="height:calc(100% - 32px);width:100%"></div>'
        '<script type="text/javascript" '
        'src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>'
        + json.dumps(cfg) + "</script></div></div>"
    )


CURRENCIES = {"USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD"}
IMPORTANCE = {
    "Hanya penting (merah)": "1",
    "Sedang dan penting": "0,1",
    "Semua berita": "-1,0,1",
}


def pair_currencies(pair):
    """EUR/USD -> EUR,USD. Emas dikaitkan ke berita USD."""
    codes = [c for c in (pair[:3], pair[4:7]) if c in CURRENCIES]
    return ",".join(codes) if codes else "USD"


def tv_calendar_html(currencies, importance, dark):
    cfg = {
        "colorTheme": "dark" if dark else "light", "isTransparent": False,
        "width": "100%", "height": "100%", "locale": "id",
        "importanceFilter": importance, "currencyFilter": currencies,
    }
    return (
        '<div style="height:500px"><div class="tradingview-widget-container" '
        'style="height:100%;width:100%"><div class="tradingview-widget-container__widget" '
        'style="height:calc(100% - 32px);width:100%"></div>'
        '<script type="text/javascript" '
        'src="https://s3.tradingview.com/external-embedding/embed-widget-events.js" async>'
        + json.dumps(cfg) + "</script></div></div>"
    )


@st.cache_data(ttl=60, show_spinner=False)
def tv_summary(pair, tf):
    """Ringkasan sinyal TradingView (library tidak resmi tradingview-ta)."""
    try:
        from tradingview_ta import Interval, TA_Handler

        _, sym, screener, exch = TV[pair]
        handler = TA_Handler(symbol=sym, screener=screener, exchange=exch,
                             interval=getattr(Interval, TV_TF[tf][1]))
        s = handler.get_analysis().summary
        return {"rec": s["RECOMMENDATION"], "buy": s["BUY"], "sell": s["SELL"],
                "neutral": s["NEUTRAL"]}
    except Exception:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def load(ticker, interval, period, resample):
    df = yf.download(ticker, interval=interval, period=period,
                     auto_adjust=False, progress=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    if resample:
        df = df.resample(resample).agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        ).dropna()
    return df


def add_indicators(df):
    c, h, l = df["Close"], df["High"], df["Low"]
    df["SMA20"] = c.rolling(20).mean()
    df["SMA50"] = c.rolling(50).mean()
    df["EMA200"] = c.ewm(span=200, adjust=False).mean()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["RSI"] = 100 - 100 / (1 + gain / loss)

    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIG"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_HIST"] = df["MACD"] - df["MACD_SIG"]

    sd = c.rolling(20).std()
    df["BB_UP"], df["BB_LO"] = df["SMA20"] + 2 * sd, df["SMA20"] - 2 * sd

    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    df["STOCH_K"] = 100 * (c - lo14) / (hi14 - lo14)
    df["STOCH_D"] = df["STOCH_K"].rolling(3).mean()

    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(alpha=1 / 14, adjust=False).mean()

    up, dn = h.diff(), -l.diff()
    plus_dm = ((up > dn) & (up > 0)) * up
    minus_dm = ((dn > up) & (dn > 0)) * dn
    pdi = 100 * plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / df["ATR"]
    mdi = 100 * minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / df["ATR"]
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    df["ADX"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    return df


def robot_signal(df):
    """Skor rule-based: tiap aturan memberi +1 (bullish), -1 (bearish) atau 0."""
    r, p = df.iloc[-1], df.iloc[-2]
    rules = []

    def add(name, score, note):
        rules.append({"Aturan": name, "Skor": score, "Keterangan": note})

    add("Tren (harga vs SMA50)", 1 if r.Close > r.SMA50 else -1,
        "Harga di atas SMA50" if r.Close > r.SMA50 else "Harga di bawah SMA50")
    add("Tren (SMA20 vs SMA50)", 1 if r.SMA20 > r.SMA50 else -1,
        "SMA20 di atas SMA50" if r.SMA20 > r.SMA50 else "SMA20 di bawah SMA50")
    add("Tren jangka panjang (EMA200)", 1 if r.Close > r.EMA200 else -1,
        "Harga di atas EMA200" if r.Close > r.EMA200 else "Harga di bawah EMA200")

    if r.RSI < 30:
        add("RSI", 1, f"Oversold ({r.RSI:.1f}), peluang pantulan naik")
    elif r.RSI > 70:
        add("RSI", -1, f"Overbought ({r.RSI:.1f}), peluang koreksi turun")
    else:
        add("RSI", 0, f"Netral ({r.RSI:.1f})")

    cross_up = p.MACD <= p.MACD_SIG and r.MACD > r.MACD_SIG
    cross_dn = p.MACD >= p.MACD_SIG and r.MACD < r.MACD_SIG
    if cross_up:
        add("MACD", 1, "Baru golden cross")
    elif cross_dn:
        add("MACD", -1, "Baru death cross")
    else:
        add("MACD", 1 if r.MACD > r.MACD_SIG else -1,
            "MACD di atas signal" if r.MACD > r.MACD_SIG else "MACD di bawah signal")

    if r.Close < r.BB_LO:
        add("Bollinger Bands", 1, "Harga menembus band bawah")
    elif r.Close > r.BB_UP:
        add("Bollinger Bands", -1, "Harga menembus band atas")
    else:
        add("Bollinger Bands", 0, "Harga di dalam band")

    if r.STOCH_K < 20 and r.STOCH_K > r.STOCH_D:
        add("Stochastic", 1, "Oversold dan K memotong D ke atas")
    elif r.STOCH_K > 80 and r.STOCH_K < r.STOCH_D:
        add("Stochastic", -1, "Overbought dan K memotong D ke bawah")
    else:
        add("Stochastic", 0, f"Netral (K={r.STOCH_K:.1f})")

    table = pd.DataFrame(rules)
    total, n = int(table["Skor"].sum()), len(table)
    ratio = total / n
    if ratio >= 0.5:
        verdict = "BUY kuat"
    elif ratio >= 0.2:
        verdict = "BUY"
    elif ratio <= -0.5:
        verdict = "SELL kuat"
    elif ratio <= -0.2:
        verdict = "SELL"
    else:
        verdict = "NETRAL"
    return table, total, n, verdict


@st.cache_data(ttl=900, show_spinner=False)
def load_news():
    """Kalender berita dari feed publik ForexFactory (tidak resmi, tanpa nilai aktual)."""
    try:
        req = urllib.request.Request("https://nfs.faireconomy.media/ff_calendar_thisweek.json",
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as f:
            df = pd.DataFrame(json.loads(f.read().decode("utf-8")))
        df["time_utc"] = pd.to_datetime(df["date"], utc=True)
        df["waktu"] = df["time_utc"].dt.tz_convert("Asia/Jakarta").dt.strftime("%d/%m %H:%M")
        return df.rename(columns={"country": "currency"})
    except Exception:
        return None


def news_risk(pair, news):
    """Risiko berita untuk pair: TINGGI bila berita merah dalam 2 jam ke depan atau 1 jam terakhir."""
    if news is None or news.empty:
        return None
    codes = pair_currencies(pair).split(",")
    ev = news[news["currency"].isin(codes) & news["impact"].isin(["High", "Medium"])].copy()
    ev["menit"] = (ev["time_utc"] - pd.Timestamp.now(tz="UTC")).dt.total_seconds() / 60
    ev = ev[(ev["menit"] > -60) & (ev["menit"] <= 24 * 60)].sort_values("time_utc")
    high = ev[ev["impact"] == "High"]
    if high["menit"].between(-60, 120).any():
        level = "TINGGI"
    elif len(high) or ev["menit"].between(-30, 120).any():
        level = "SEDANG"
    else:
        level = "RENDAH"
    return {"level": level, "events": ev}


def candle_pattern(df):
    r, p = df.iloc[-1], df.iloc[-2]
    rng, body = r.High - r.Low, abs(r.Close - r.Open)
    if rng == 0:
        return None
    lower, upper = min(r.Open, r.Close) - r.Low, r.High - max(r.Open, r.Close)
    if p.Close < p.Open and r.Close > r.Open and r.Close >= p.Open and r.Open <= p.Close:
        return ("Bullish engulfing (pembalikan naik)", 1)
    if p.Close > p.Open and r.Close < r.Open and r.Close <= p.Open and r.Open >= p.Close:
        return ("Bearish engulfing (pembalikan turun)", -1)
    if lower >= 2 * max(body, rng * 0.05) and upper <= rng * 0.25:
        return ("Hammer / pin bar (tekanan jual ditolak)", 1)
    if upper >= 2 * max(body, rng * 0.05) and lower <= rng * 0.25:
        return ("Shooting star (tekanan beli ditolak)", -1)
    if body < rng * 0.1:
        return ("Doji (pasar ragu-ragu)", 0)
    return None


@st.cache_data(ttl=60, show_spinner=False)
def mtf_scores(pair):
    """Skor robot di 1 jam, 4 jam, dan harian untuk melihat keselarasan timeframe."""
    out = {}
    for label in ("1 jam", "4 jam", "Harian"):
        try:
            iv, per, rs = TIMEFRAMES[label]
            d = add_indicators(load(PAIRS[pair], iv, per, rs)).dropna()
            _, t, _, v = robot_signal(d)
            out[label] = (t, v)
        except Exception:
            pass
    return out


FX_MAJORS = {
    "EURUSD=X": ("EUR", "USD"), "GBPUSD=X": ("GBP", "USD"), "USDJPY=X": ("USD", "JPY"),
    "AUDUSD=X": ("AUD", "USD"), "USDCAD=X": ("USD", "CAD"), "USDCHF=X": ("USD", "CHF"),
    "NZDUSD=X": ("NZD", "USD"),
}
DXY_TICKER = "DX-Y.NYB"


def strength_from_close(close, lookback=20):
    """Kekuatan tiap mata uang (% perubahan rata-rata) dari 7 pair mayor."""
    if len(close) <= lookback:
        return {}
    ret = close.iloc[-1] / close.iloc[-1 - lookback] - 1
    acc = {}
    for t, (base, quote) in FX_MAJORS.items():
        if t in ret.index and pd.notna(ret[t]):
            acc.setdefault(base, []).append(ret[t] * 100)
            acc.setdefault(quote, []).append(-ret[t] * 100)
    return {c: sum(v) / len(v) for c, v in acc.items()}


@st.cache_data(ttl=60, show_spinner=False)
def fx_context(tf):
    """Indeks kekuatan mata uang buatan sendiri + tren DXY, pada timeframe terpilih."""
    try:
        interval, period, resample = TIMEFRAMES[tf]
        tickers = list(FX_MAJORS) + [DXY_TICKER]
        raw = yf.download(tickers, interval=interval, period=period, auto_adjust=False,
                          progress=False, group_by="ticker")
        frames = {}
        for t in tickers:
            try:
                d = raw[t][["Open", "High", "Low", "Close"]].dropna()
            except Exception:
                continue
            if d.empty:
                continue
            if d.index.tz is not None:
                d.index = d.index.tz_convert("UTC")
            if resample:
                d = d.resample(resample).agg(
                    {"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
            frames[t] = d
        close = pd.DataFrame({t: frames[t]["Close"] for t in FX_MAJORS if t in frames}).ffill()
        strength = strength_from_close(close)
        if not strength:
            return None
        dxy = None
        if DXY_TICKER in frames:
            d = add_indicators(frames[DXY_TICKER].copy()).dropna()
            if len(d) >= 30:
                _, tot, _, _ = robot_signal(d)
                dxy = (tot, "Menguat" if tot > 0 else "Melemah" if tot < 0 else "Netral")
        return {"strength": strength, "dxy": dxy}
    except Exception:
        return None


def currency_factor(pair, d, fx):
    """Selaras tidaknya arah sinyal dengan kekuatan mata uang dan DXY. Return (poin, maks, catatan)."""
    if not fx or not d:
        return None
    s = fx["strength"]
    ranks = {c: i for i, c in enumerate(sorted(s, key=s.get, reverse=True))}  # 0 = terkuat
    base, quote = pair[:3], pair[4:7]
    pts, mx, notes = 0, 0, []
    if base in ranks and quote in ranks:
        gap = (ranks[quote] - ranks[base]) * d
        pts += 15 if gap >= 3 else 8 if gap >= 1 else 0
        mx += 15
        notes.append(f"{base} peringkat {ranks[base] + 1} dan {quote} peringkat {ranks[quote] + 1} dari {len(ranks)}")
    elif quote in ranks:  # emas: lihat kekuatan USD
        sig = (ranks[quote] - (len(ranks) - 1) / 2) * d
        pts += 15 if sig >= 2 else 8 if sig >= 0 else 0
        mx += 15
        notes.append(f"USD peringkat {ranks[quote] + 1} dari {len(ranks)}")
    if fx.get("dxy") and "USD" in (base, quote):
        dd = (fx["dxy"][0] > 0) - (fx["dxy"][0] < 0)
        implied = -dd if quote == "USD" else dd
        pts += 10 if dd and implied == d else 4 if dd == 0 else 0
        mx += 10
        notes.append(f"DXY {fx['dxy'][1].lower()}")
    return (pts, mx, "; ".join(notes)) if mx else None


def confidence(total, n, mtf, adx, tv, tv_dir, nr, fxf=None):
    d = 1 if total > 0 else -1 if total < 0 else 0
    f = [("Kekuatan sinyal robot", round(min(abs(total) / n / 0.6, 1) * 30), 30, f"skor {total}/{n}")]
    if mtf and d:
        same = sum(1 for t, _ in mtf.values() if (t > 0) - (t < 0) == d)
        f.append(("Searah di 1j / 4j / harian", round(25 * same / len(mtf)), 25,
                  f"{same} dari {len(mtf)} timeframe searah"))
    else:
        f.append(("Searah di 1j / 4j / harian", 0, 25, "tidak ada arah atau data kosong"))
    f.append(("Kekuatan tren (ADX)", 10 if adx >= 25 else 5 if adx >= 20 else 0, 10, f"ADX {adx:.0f}"))
    if tv is None:
        f.append(("Sinyal TradingView", 5, 10, "tidak tersedia (nilai tengah)"))
    elif d and tv_dir == d:
        f.append(("Sinyal TradingView", 10, 10, "searah dengan robot"))
    elif tv_dir == 0:
        f.append(("Sinyal TradingView", 3, 10, "netral"))
    else:
        f.append(("Sinyal TradingView", 0, 10, "berlawanan atau robot netral"))
    if fxf:
        f.append(("Selaras DXY / indeks mata uang", round(fxf[0] / fxf[1] * 25), 25, fxf[2]))
    else:
        f.append(("Selaras DXY / indeks mata uang", 12, 25, "data tidak tersedia atau tanpa arah (nilai tengah)"))
    pts = sum(x[1] for x in f)
    if nr:
        pen = {"TINGGI": -25, "SEDANG": -10, "RENDAH": 0}[nr["level"]]
        f.append(("Filter berita", pen, 0, f"risiko berita {nr['level'].lower()}"))
        pts += pen
    else:
        f.append(("Filter berita", 0, 0, "data berita tidak tersedia"))
    score = max(0, min(100, pts))
    if nr and nr["level"] == "TINGGI":
        dec = "HINDARI (ada berita penting)"
    elif d == 0:
        dec = "TUNGGU"
    elif score >= 65:
        dec = "LAYAK ENTRI"
    elif score >= 45:
        dec = "HATI-HATI"
    else:
        dec = "TUNGGU"
    return score, f, dec


def pip_size(pair):
    if "XAU" in pair:
        return 0.1
    return 0.01 if "JPY" in pair else 0.0001


def trade_levels(df, total, sl_mult=1.0, rr=1.5):
    """Level entri, SL, TP dan support/resistance untuk digambar di grafik."""
    r = df.iloc[-1]
    price, atr = r.Close, r.ATR
    sup, res = key_levels(df, price)
    lv = {"sup": sup, "res": res, "dir": None}
    if total > 0:
        risk = sl_mult * atr
        lv.update(dir="BUY", entry=price, sl=price - risk,
                  tp1=price + rr * risk, tp2=price + (rr + 1) * risk)
    elif total < 0:
        risk = sl_mult * atr
        lv.update(dir="SELL", entry=price, sl=price + risk,
                  tp1=price - rr * risk, tp2=price - (rr + 1) * risk)
    return lv


def usd_per_quote(pair, price):
    """Konversi mata uang kuotasi (belakang) ke USD."""
    if pair[4:7] == "USD":
        return 1.0
    if pair[:3] == "USD":
        return 1.0 / price
    try:  # silang JPY (EUR/JPY, GBP/JPY)
        return 1.0 / float(load("USDJPY=X", "1d", "5d", None)["Close"].iloc[-1])
    except Exception:
        return 1.0 / 150.0


def lot_math(pair, lv, price, modal, kurs, risk_pct, units):
    """Hitung lot untuk akun cent. Semua nilai uang dalam cent (USC)."""
    pip = pip_size(pair)
    sl_pips = abs(lv["entry"] - lv["sl"]) / pip
    pv = units * pip * usd_per_quote(pair, price) * 100  # cent per pip per 1 lot
    bal = modal / kurs * 100
    risk = bal * risk_pct / 100
    raw = risk / (sl_pips * pv)
    lot = max(0.01, math.floor(raw * 100) / 100)
    return {
        "bal": bal, "risk": risk, "raw": raw, "lot": lot, "sl_pips": sl_pips,
        "pv001": pv * 0.01,
        "loss": lot * sl_pips * pv,
        "gain1": lot * abs(lv["tp1"] - lv["entry"]) / pip * pv,
        "gain2": lot * abs(lv["tp2"] - lv["entry"]) / pip * pv,
    }


def lot_panel(pair, lv, price, modal, kurs, risk_pct, units):
    if "XAU" in pair:
        st.info("Spesifikasi kontrak emas berbeda di tiap broker, jadi kalkulator lot hanya untuk pair forex.")
        return
    if not lv["dir"]:
        st.info("Sinyal netral: tidak ada entri, jadi tidak ada lot yang dihitung.")
        return
    m = lot_math(pair, lv, price, modal, kurs, risk_pct, units)
    rp = lambda cent: cent / 100 * kurs
    c1, c2, c3 = st.columns(3)
    c1.metric("Lot disarankan", f"{m['lot']:.2f}")
    c2.metric("Rugi jika kena SL", f"{m['loss']:.1f} cent", f"≈ Rp{rp(m['loss']):,.0f} ({m['loss'] / m['bal'] * 100:.1f}% modal)".replace(",", "."), delta_color="off")
    c3.metric("Modal", f"{m['bal']:.0f} cent", f"≈ Rp{modal:,.0f}".replace(",", "."), delta_color="off")
    st.write(f"Untung jika kena Target 1: **{m['gain1']:.1f} cent** (≈ Rp{rp(m['gain1']):,.0f}) · "
             f"Target 2: **{m['gain2']:.1f} cent** (≈ Rp{rp(m['gain2']):,.0f})".replace(",", "."))
    if m["raw"] < 0.01:
        pct = m["loss"] / m["bal"] * 100
        msg = (f"Lot minimum 0.01 sudah berisiko {pct:.1f}% modal, lebih besar dari batas {risk_pct:g}% Anda. "
               "Perkecil jarak SL (timeframe lebih kecil atau slider SL), atau tambah modal.")
        (st.error if pct > 5 else st.warning)(msg)
    st.caption(f"Jarak SL {m['sl_pips']:.0f} pip. Nilai 1 pip pada 0.01 lot ≈ {m['pv001']:.3f} cent. "
               "Cocokkan dengan spesifikasi pair di MT4/MT5 broker Anda (Contract size); kalau beda, ubah pilihan "
               f"Unit per 1 lot di sidebar. Aturan umum: berhenti trading hari itu bila rugi mencapai "
               f"3× risiko ({3 * m['risk']:.0f} cent). Margin dan spread belum dihitung.")


def build_chart(df, name, lv=None):
    d = df.tail(150)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2],
                        vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=d.index, open=d.Open, high=d.High, low=d.Low,
                                 close=d.Close, name=name), row=1, col=1)
    for col, color in [("SMA20", "#f59e0b"), ("SMA50", "#3b82f6"), ("EMA200", "#a855f7"),
                       ("BB_UP", "#94a3b8"), ("BB_LO", "#94a3b8")]:
        fig.add_trace(go.Scatter(x=d.index, y=d[col], name=col, line=dict(width=1, color=color)),
                      row=1, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d.RSI, name="RSI", line=dict(color="#0e8f7e")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dot", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", row=2, col=1)
    fig.add_trace(go.Bar(x=d.index, y=d.MACD_HIST, name="MACD hist"), row=3, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d.MACD, name="MACD", line=dict(width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=d.index, y=d.MACD_SIG, name="Signal", line=dict(width=1)), row=3, col=1)
    if lv:
        step = d.index[-1] - d.index[-2]
        x0, x1 = d.index[-1] - step * 8, d.index[-1] + step * 14
        ys = [d.High.max(), d.Low.min()]

        def hline(y, label, color, dash="solid"):
            fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y, row=1, col=1,
                          line=dict(color=color, width=1.6, dash=dash))
            fig.add_annotation(x=x1, y=y, text=f"{label} {y:.5f}", showarrow=False,
                               xanchor="left", font=dict(color=color, size=11), row=1, col=1)
            ys.append(y)

        hline(lv["sup"][0], "Support", "#64748b", "dot")
        hline(lv["res"][0], "Resistance", "#64748b", "dot")
        if lv["dir"]:
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=lv["entry"], y1=lv["sl"], row=1, col=1,
                          fillcolor="rgba(239,68,68,0.20)", line_width=0)
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=lv["entry"], y1=lv["tp2"], row=1, col=1,
                          fillcolor="rgba(34,197,94,0.18)", line_width=0)
            hline(lv["entry"], f"ENTRI {lv['dir']}", "#3b82f6")
            hline(lv["sl"], "SL", "#ef4444")
            hline(lv["tp1"], "TP1", "#22c55e")
            hline(lv["tp2"], "TP2", "#16a34a")
            is_buy = lv["dir"] == "BUY"
            fig.add_trace(go.Scatter(
                x=[d.index[-1]], y=[lv["entry"]], mode="markers+text", showlegend=False,
                text=[f"<b>{lv['dir']} {lv['entry']:.5f}</b>"],
                textposition="bottom center" if is_buy else "top center",
                textfont=dict(color="#3b82f6", size=13),
                hovertemplate=f"{lv['dir']} @ %{{y:.5f}}<extra></extra>",
                marker=dict(symbol="triangle-up" if is_buy else "triangle-down",
                            size=15, color="#3b82f6")), row=1, col=1)
        pad = (max(ys) - min(ys)) * 0.05
        fig.update_yaxes(range=[min(ys) - pad, max(ys) + pad], row=1, col=1)
        fig.update_xaxes(range=[d.index[0], x1 + step * 10])
    fig.update_layout(height=720, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.02))
    return fig


def key_levels(df, price, window=5, lookback=120):
    """Cari support/resistance dari swing high/low terdekat."""
    d = df.tail(lookback)
    highs, lows = d["High"], d["Low"]
    sh, sl = [], []
    for i in range(window, len(d) - window):
        if highs.iloc[i] == highs.iloc[i - window:i + window + 1].max():
            sh.append(highs.iloc[i])
        if lows.iloc[i] == lows.iloc[i - window:i + window + 1].min():
            sl.append(lows.iloc[i])
    res = sorted(x for x in sh if x > price) or [d["High"].max()]
    sup = sorted((x for x in sl if x < price), reverse=True) or [d["Low"].min()]
    return sup[:2], res[:2]


def auto_analysis(pair, tf, df, table, verdict, total, n, ctx=None):
    """Mesin analisa lokal: menyusun narasi dari indikator, tanpa API."""
    r, p = df.iloc[-1], df.iloc[-2]
    price, atr = r.Close, r.ATR
    slm, rr = (ctx["sl_mult"], ctx["rr"]) if ctx else (1.0, 1.5)
    lv_ = trade_levels(df, total, slm, rr)
    dist = abs(price - lv_["sl"]) / pip_size(pair) if lv_["dir"] else 0
    sup, res = key_levels(df, price)

    # Bias arah
    if total >= 2:
        bias = "bullish (cenderung naik)"
    elif total <= -2:
        bias = "bearish (cenderung turun)"
    else:
        bias = "sideways / belum jelas"

    # Tren
    trend = []
    trend.append("harga di atas SMA50" if price > r.SMA50 else "harga di bawah SMA50")
    trend.append("di atas EMA200 (tren besar naik)" if price > r.EMA200 else "di bawah EMA200 (tren besar turun)")
    # Momentum
    mom = []
    if r.RSI > 70:
        mom.append(f"RSI {r.RSI:.0f} overbought, rawan koreksi")
    elif r.RSI < 30:
        mom.append(f"RSI {r.RSI:.0f} oversold, rawan pantulan")
    else:
        mom.append(f"RSI {r.RSI:.0f} netral")
    mom.append("histogram MACD menguat" if r.MACD_HIST > p.MACD_HIST else "histogram MACD melemah")
    # Volatilitas
    atr_avg = df["ATR"].tail(50).mean()
    if atr > 1.3 * atr_avg:
        vol = "volatilitas tinggi (ATR di atas rata-rata), gunakan stop loss lebih longgar dan lot lebih kecil"
    elif atr < 0.7 * atr_avg:
        vol = "volatilitas rendah (ATR di bawah rata-rata), pasar cenderung sempit dan rawan breakout"
    else:
        vol = "volatilitas normal"

    fmt = lambda x: f"{x:.5f}"
    out = [f"### Analisa {pair} ({tf})",
           f"**Bias: {bias}** | Sinyal robot: **{verdict}** (skor {total}/{n})", "",
           f"**Kondisi pasar.** {trend[0].capitalize()} dan {trend[1]}. {mom[0]}, {mom[1]}. "
           f"Saat ini {vol}.", "",
           f"**Level penting.** Support terdekat: {', '.join(fmt(x) for x in sup)}. "
           f"Resistance terdekat: {', '.join(fmt(x) for x in res)}.", ""]

    if total >= 2:
        out += [f"**Skenario utama (buy).** Entri dekat {fmt(price)} atau saat harga kembali menguji "
                f"{fmt(sup[0])}. Stop loss {fmt(lv_['sl'])} (jarak ≈ {dist:.0f} pip). "
                f"Target 1 {fmt(lv_['tp1'])} (1:{rr:g}), target 2 {fmt(lv_['tp2'])} (1:{rr + 1:g}).",
                f"**Skenario alternatif.** Jika harga ditutup di bawah {fmt(sup[0])}, ide buy batal dan "
                f"harga bisa melanjutkan koreksi ke {fmt(sup[-1])}."]
    elif total <= -2:
        out += [f"**Skenario utama (sell).** Entri dekat {fmt(price)} atau saat harga memantul ke "
                f"{fmt(res[0])}. Stop loss {fmt(lv_['sl'])} (jarak ≈ {dist:.0f} pip). "
                f"Target 1 {fmt(lv_['tp1'])} (1:{rr:g}), target 2 {fmt(lv_['tp2'])} (1:{rr + 1:g}).",
                f"**Skenario alternatif.** Jika harga ditutup di atas {fmt(res[0])}, ide sell batal dan "
                f"harga bisa naik ke {fmt(res[-1])}."]
    else:
        out += [f"**Skenario utama.** Harga cenderung bergerak di antara {fmt(sup[0])} dan {fmt(res[0])}. "
                "Lebih aman menunggu breakout yang jelas, atau trading dari tepi range dengan stop loss ketat.",
                f"**Skenario alternatif.** Breakout di atas {fmt(res[0])} membuka peluang buy, "
                f"sedangkan tembus ke bawah {fmt(sup[0])} membuka peluang sell."]

    if ctx:
        adx = ctx["adx"]
        if adx >= 25:
            regime = "tren kuat, strategi mengikuti arah lebih masuk akal"
        elif adx >= 20:
            regime = "tren mulai terbentuk tapi belum kuat"
        else:
            regime = "pasar cenderung ranging, sinyal tren rawan palsu"
        out += ["", f"**Kekuatan tren.** ADX {adx:.0f}: {regime}."]
        if ctx["mtf"]:
            parts = ", ".join(f"{k}: {v[1]}" for k, v in ctx["mtf"].items())
            d = 1 if total > 0 else -1 if total < 0 else 0
            same = sum(1 for t, _ in ctx["mtf"].values() if (t > 0) - (t < 0) == d) if d else 0
            note = "timeframe selaras, keyakinan lebih tinggi" if d and same == len(ctx["mtf"]) \
                else "timeframe belum selaras, perlu hati-hati"
            out += ["", f"**Multi-timeframe.** {parts}. Hasilnya: {note}."]
        if ctx["pat"]:
            out += ["", f"**Pola candle terakhir.** {ctx['pat'][0]}."]
        if ctx.get("fx_note"):
            out += ["", f"**Indeks mata uang dan DXY.** {ctx['fx_note']}"]
        nr = ctx["nr"]
        if nr is None:
            out += ["", "**Berita.** Data berita otomatis tidak tersedia, cek kalender secara manual."]
        else:
            ev = nr["events"].head(3)
            daftar = "; ".join(f"{r.currency} {r.title} ({r.waktu} WIB, {r.impact})" for r in ev.itertuples()) \
                or "tidak ada berita sedang/penting dalam 24 jam"
            out += ["", f"**Berita.** Risiko {nr['level'].lower()}. Terdekat: {daftar}."]
    out += ["", "**Risiko.** Rilis berita ekonomi (NFP, suku bunga, CPI) dapat membalik arah dengan cepat. "
                "Hindari entri tepat sebelum berita besar, batasi risiko per transaksi maksimal 1-2% modal, "
                "dan konfirmasi dengan timeframe yang lebih besar."]
    if ctx:
        out += ["", f"**Kesimpulan.** Skor keyakinan {ctx['score']}/100, keputusan: **{ctx['decision']}**."]
    return "\n\n".join(out)


# ---------------- UI ----------------
st.title("Robot analisa forex")
st.caption("Grafik real-time dari TradingView. Robot memakai data Yahoo Finance (bisa tertunda). Bukan saran investasi.")

with st.sidebar:
    pair = st.selectbox("Pair", list(PAIRS))
    tf = st.selectbox("Timeframe", list(TIMEFRAMES), index=1)
    dark = st.toggle("Tema grafik gelap", value=True)
    sl_mult = st.slider("Jarak stop loss (× ATR)", 0.5, 3.0, 1.0, 0.1,
                        help="Makin kecil, SL makin dekat. ATR di timeframe besar nilainya besar.")
    rr = st.slider("Target risk:reward", 1.0, 4.0, 2.0, 0.5,
                   help="Target 1 = 1:RR, Target 2 = 1:(RR+1).")
    st.markdown("**Modal dan risiko (akun cent)**")
    modal = st.number_input("Modal (Rp)", min_value=10000, value=100000, step=10000)
    kurs = st.number_input("Kurs USD/IDR", min_value=10000, value=17900, step=50,
                           help="Isi sesuai kurs hari ini.")
    risk_pct = st.slider("Risiko per transaksi (%)", 0.5, 3.0, 1.0, 0.5)
    units = st.selectbox("Unit per 1 lot (akun cent)", [1000, 10000, 100000],
                         format_func=lambda x: f"{x:,}".replace(",", "."),
                         help="Cek Contract size pair di MT4/MT5 broker Anda. Tiap broker bisa berbeda.")
    news_level = st.selectbox("Tingkat berita", list(IMPORTANCE), index=1)
    news_all = st.toggle("Berita semua mata uang", value=False,
                         help="Matikan untuk hanya menampilkan berita mata uang pair terpilih.")
    auto = st.toggle("Refresh otomatis tiap 1 menit", value=False,
                     help="Hanya memperbarui panel robot; grafik TradingView sudah real-time.")
    st.button("Muat ulang data", on_click=st.cache_data.clear)

st.subheader("Grafik real-time (TradingView)")
components.html(tv_chart_html(TV[pair][0], TV_TF[tf][0], dark), height=640)

st.subheader("Jadwal berita ekonomi")
codes = ",".join(sorted(CURRENCIES)) if news_all else pair_currencies(pair)
st.caption(f"Mata uang: {codes.replace(',', ', ')}. Waktu mengikuti zona waktu browser Anda. "
           "Hindari membuka posisi menjelang berita penting.")
components.html(tv_calendar_html(codes, IMPORTANCE[news_level], dark), height=520)


def analysis_panel():
    interval, period, resample = TIMEFRAMES[tf]
    with st.spinner("Mengambil data..."):
        data = load(PAIRS[pair], interval, period, resample)
    if data.empty or len(data) < 60:
        st.error("Data robot tidak tersedia atau terlalu sedikit. Pasar mungkin tutup; coba timeframe lain.")
        return

    data = add_indicators(data).dropna()
    table, total, n, verdict = robot_signal(data)
    last = data.iloc[-1]

    st.subheader("Sinyal robot")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Harga terakhir (Yahoo)", f"{last.Close:.5f}")
    c2.metric(f"Sinyal robot (skor {total}/{n})", verdict)
    c3.metric("RSI (14)", f"{last.RSI:.1f}")
    c4.metric("ATR (14)", f"{last.ATR:.5f}")

    tv = tv_summary(pair, tf)
    tv_dir = 0 if tv is None else (1 if "BUY" in tv["rec"] else -1 if "SELL" in tv["rec"] else 0)
    mtf = mtf_scores(pair)
    news = load_news()
    nr = news_risk(pair, news)
    fx = fx_context(tf)
    d_dir = 1 if total > 0 else -1 if total < 0 else 0
    fxf = currency_factor(pair, d_dir, fx)
    fx_note = ""
    if fx:
        order = sorted(fx["strength"], key=fx["strength"].get, reverse=True)
        fx_note = f"Terkuat {order[0]}, terlemah {order[-1]}."
        if fx["dxy"]:
            fx_note += f" DXY (indeks dolar) {fx['dxy'][1].lower()}."
        if fxf:
            fx_note += f" {fxf[2]}."
    score, factors, decision = confidence(total, n, mtf, last.ADX, tv, tv_dir, nr, fxf)
    ctx = {"adx": last.ADX, "mtf": mtf, "pat": candle_pattern(data), "nr": nr,
           "score": score, "decision": decision, "sl_mult": sl_mult, "rr": rr, "fx_note": fx_note}

    st.subheader("Keputusan akhir")
    k1, k2 = st.columns([1, 2])
    k1.metric("Skor keyakinan", f"{score}/100")
    k1.metric("Keputusan", decision)
    k2.dataframe(pd.DataFrame(factors, columns=["Faktor", "Poin", "Maks", "Keterangan"]),
                 hide_index=True, use_container_width=True)
    if nr is None:
        st.info("Data berita otomatis tidak tersedia (offline atau dibatasi). Cek kalender di atas secara manual.")
    else:
        msg = f"Risiko berita {nr['level'].lower()} untuk {pair}."
        {"TINGGI": st.error, "SEDANG": st.warning, "RENDAH": st.success}[nr["level"]](msg)
        if not nr["events"].empty:
            ev = nr["events"][["waktu", "currency", "title", "impact", "forecast", "previous"]]
            ev.columns = ["Waktu (WIB)", "Mata uang", "Berita", "Dampak", "Forecast", "Sebelumnya"]
            st.dataframe(ev, hide_index=True, use_container_width=True)

    st.subheader("Kekuatan mata uang dan DXY")
    if fx is None:
        st.info("Data indeks mata uang tidak tersedia saat ini. Faktor ini memakai nilai tengah di skor keyakinan.")
    else:
        s = fx["strength"]
        order = sorted(s, key=s.get, reverse=True)
        mark = {pair[:3], pair[4:7]}
        fig_fx = go.Figure(go.Bar(
            x=order, y=[s[c] for c in order],
            marker_color=["#3b82f6" if c in mark else "#94a3b8" for c in order],
            text=[f"{s[c]:+.2f}%" for c in order], textposition="outside"))
        fig_fx.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10),
                             yaxis_title="Perubahan 20 candle (%)")
        ca, cb = st.columns([3, 2])
        ca.plotly_chart(fig_fx, use_container_width=True)
        if fx["dxy"]:
            cb.metric("Indeks dolar (DXY)", fx["dxy"][1], f"skor robot {fx['dxy'][0]}", delta_color="off")
        else:
            cb.info("Data DXY tidak tersedia.")
        cb.write(f"Terkuat: **{order[0]}** · Terlemah: **{order[-1]}**")
        if fxf:
            cb.caption(f"Untuk {pair}: {fxf[2]}.")
        else:
            cb.caption("Sinyal netral, jadi keselarasan tidak dinilai.")
        st.caption("Biru = mata uang pair terpilih. Indeks dibuat dari 7 pair mayor Yahoo (bukan indeks resmi), "
                   "dengan acuan timeframe yang dipilih.")

    lv = trade_levels(data, total, sl_mult, rr)
    st.subheader("Grafik level entri")
    if lv["dir"]:
        st.caption(f"Zona merah = risiko (entri ke SL), zona hijau = target (entri ke TP2). Arah: {lv['dir']}.")
    else:
        st.caption("Sinyal netral: tidak ada level entri. Hanya support dan resistance terdekat yang ditampilkan.")
    st.plotly_chart(build_chart(data, pair, lv), use_container_width=True)

    st.subheader("Perbandingan dengan sinyal TradingView")
    if tv is None:
        st.info("Ringkasan TradingView tidak tersedia (library tradingview-ta belum terpasang "
                "atau sedang dibatasi). Robot tetap berjalan.")
    else:
        rec = tv["rec"]
        robot_dir = 1 if total > 0 else -1 if total < 0 else 0
        tv_dir = 1 if "BUY" in rec else -1 if "SELL" in rec else 0
        a, b = st.columns(2)
        a.metric("Robot (data Yahoo)", verdict)
        b.metric("TradingView", rec.replace("_", " "))
        st.caption(f"TradingView: {tv['buy']} indikator buy, {tv['neutral']} netral, {tv['sell']} sell.")
        if robot_dir == tv_dir != 0:
            st.success("Kedua sumber searah, sinyal lebih meyakinkan (tetap bukan jaminan).")
        elif robot_dir == 0 or tv_dir == 0:
            st.warning("Salah satu sumber netral. Lebih baik menunggu konfirmasi.")
        else:
            st.error("Kedua sumber berlawanan arah. Hati-hati, sebaiknya tunggu atau cek timeframe lain.")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Detail aturan robot")
        st.dataframe(table, hide_index=True, use_container_width=True)
    with right:
        st.subheader("Level saran (berdasarkan ATR)")
        if lv["dir"]:
            st.write(f"Arah: **{lv['dir']}** · Entri ≈ {lv['entry']:.5f}")
            st.write(f"Stop loss ≈ {lv['sl']:.5f} (jarak ≈ {abs(lv['entry'] - lv['sl']) / pip_size(pair):.0f} pip)")
            st.write(f"Target 1 (1:{rr:g}) ≈ {lv['tp1']:.5f} · Target 2 (1:{rr + 1:g}) ≈ {lv['tp2']:.5f}")
        else:
            st.write("Sinyal netral: tidak ada arah yang jelas, lebih baik menunggu.")
        st.caption(f"SL = {sl_mult:g}× ATR. Ubah jarak SL dan target risk:reward di sidebar.")

    st.subheader("Kalkulator lot (akun cent)")
    lot_panel(pair, lv, last.Close, modal, kurs, risk_pct, units)

    st.divider()
    st.subheader("Analisa otomatis")
    st.markdown(auto_analysis(pair, tf, data, table, verdict, total, n, ctx))


st.fragment(analysis_panel, run_every=60 if auto else None)()
