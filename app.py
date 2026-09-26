import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import accuracy_score
from datetime import date

# ---------- PAGE ----------
st.set_page_config(page_title="NEXUS AI",page_icon="⚡",layout="wide")

st.markdown("""
<style>
.stApp{background:#050812;color:white}
h1,h2,h3{color:#00e5ff}
.card{background:#0c1424;padding:15px;border-radius:15px;
border:1px solid #17304c;margin:4px}
.big{font-size:27px;font-weight:bold}
</style>
""",unsafe_allow_html=True)

# ---------- STOCKS ----------
STOCKS={
"TCS":"TCS.NS","Infosys":"INFY.NS","Reliance":"RELIANCE.NS",
"HDFC Bank":"HDFCBANK.NS","ICICI Bank":"ICICIBANK.NS",
"Apple":"AAPL","Microsoft":"MSFT","Google":"GOOGL",
"Amazon":"AMZN","Tesla":"TSLA","NVIDIA":"NVDA","Meta":"META"
}

def card(a,b):
    st.markdown(
        f'<div class="card"><b>{a}</b><div class="big">{b}</div></div>',
        unsafe_allow_html=True)

def graph(x,y,title):
    fig,ax=plt.subplots(figsize=(10,3))
    ax.plot(x,y)
    ax.set_title(title)
    ax.grid(alpha=.2)
    st.pyplot(fig)

# ---------- FEATURES ----------
def features(d):
    d=d.copy()
    d["MA20"]=d.Close.rolling(20).mean()
    d["MA50"]=d.Close.rolling(50).mean()
    ch=d.Close.diff()
    d["RSI"]=100-100/(1+ch.clip(lower=0).rolling(14).mean()/
                     ch.clip(upper=0).abs().rolling(14).mean())
    d["MACD"]=d.Close.ewm(12).mean()-d.Close.ewm(26).mean()
    d["MOM"]=d.Close.pct_change(5)
    d["VOL"]=d.Close.pct_change().rolling(20).std()
    return d

# ---------- TITLE ----------
st.title("⚡ NEXUS AI")
st.caption("AI MARKET INTELLIGENCE COMMAND CENTER")

name=st.selectbox("🔎 Search Stock",list(STOCKS))
sym=STOCKS[name]

# ---------- DATA ----------
d=yf.download(sym,period="2y",auto_adjust=True,progress=False)

if isinstance(d.columns,pd.MultiIndex):
    d.columns=d.columns.get_level_values(0)

d=features(d)

cols=["Open","High","Low","Close","Volume",
      "MA20","MA50","RSI","MACD","MOM","VOL"]

# Correct next-day target
d["Target"]=(d.Close.shift(-1)>d.Close).astype(float)
d.loc[d.Close.shift(-1).isna(),"Target"]=np.nan

m=d.dropna(subset=cols+["Target"])

X=m[cols]
Y=m.Target

cut=int(len(m)*.8)

# ---------- AI MODEL ----------
model=RandomForestClassifier(
    n_estimators=120,
    random_state=1)

model.fit(X.iloc[:cut],Y.iloc[:cut])

test=model.predict(X.iloc[cut:])
accuracy=accuracy_score(Y.iloc[cut:],test)*100

latest=d.iloc[-1]
yest=d.iloc[-2]
before_yest=d.iloc[-3]

pred=model.predict(
    latest[cols].to_frame().T)[0]

confidence=model.predict_proba(
    latest[cols].to_frame().T)[0].max()

next_direction="📈 RISE" if pred else "📉 FALL"

# ---------- PRICE MODEL ----------
future_target=m.Close.shift(-1)

reg=RandomForestRegressor(
    n_estimators=100,
    random_state=1)

reg.fit(
    X.iloc[:cut],
    future_target.iloc[:cut])

next_price=reg.predict(
    latest[cols].to_frame().T)[0]

# ---------- ACTUAL MOVEMENTS ----------
yest_change=(yest.Close/before_yest.Close-1)*100
today_change=(latest.Close/yest.Close-1)*100

yest_direction="📈 RISE" if yest_change>0 else "📉 FALL"
today_direction="📈 RISE" if today_change>0 else "📉 FALL"

# ---------- SESSION LABEL ----------
if latest.name.date()==date.today():
    latest_label="TODAY"
else:
    latest_label="LATEST SESSION"

# ==================================================
# AI CORE
# ==================================================

st.header("🧠 AI CORE")

a,b,c,e=st.columns(4)

with a:
    card("LATEST PRICE",f"{latest.Close:,.2f}")

with b:
    card(latest_label,f"{today_direction} {today_change:.2f}%")

with c:
    card("NEXT SESSION",next_direction)

with e:
    card("CONFIDENCE",f"{confidence*100:.1f}%")

# ==================================================
# PREDICTION CENTER
# ==================================================

st.header("🎯 PREDICTION CENTER")

a,b,c=st.columns(3)

with a:
    card("YESTERDAY",yest_direction)
    st.write(f"Price: {yest.Close:,.2f}")
    st.write(f"Change: {yest_change:.2f}%")

with b:
    card(latest_label,today_direction)
    st.write(f"Price: {latest.Close:,.2f}")
    st.write(f"Change: {today_change:.2f}%")

with c:
    card("NEXT SESSION",next_direction)
    st.write(f"Predicted: {next_price:,.2f}")
    st.write(f"Confidence: {confidence*100:.1f}%")

# ==================================================
# FUTURE DATE
# ==================================================

st.header("🔮 FUTURE DATE PREDICTION")

target=pd.Timestamp(
    st.date_input("Select Future Date"))

# ---------- WEEKEND ----------
if target.weekday()>=5:

    past=d[d.index.normalize()<target]

    if len(past):

        actual=past.iloc[-1]

        st.warning(
            f"⚠️ No trading on {target.date()}. "
            f"Showing last trading day: {actual.name.date()}."
        )

        if len(past)>1:

            previous=past.iloc[-2]
            change=(actual.Close/previous.Close-1)*100
            direction="📈 RISE" if change>0 else "📉 FALL"

            a,b,c=st.columns(3)

            with a:
                card("LAST TRADING DAY",direction)

            with b:
                card("ACTUAL PRICE",
                     f"{actual.Close:,.2f}")

            with c:
                card("ACTUAL CHANGE",
                     f"{change:.2f}%")

# ---------- PAST / CURRENT ----------
elif target<=latest.name.normalize():

    past=d[d.index.normalize()<=target]

    if len(past):

        actual=past.iloc[-1]

        if len(past)>1:

            previous=past.iloc[-2]

            change=(actual.Close/
                    previous.Close-1)*100

            direction="📈 RISE" if change>0 else "📉 FALL"

            a,b,c=st.columns(3)

            with a:
                card("ACTUAL DIRECTION",direction)

            with b:
                card("ACTUAL PRICE",
                     f"{actual.Close:,.2f}")

            with c:
                card("ACTUAL CHANGE",
                     f"{change:.2f}%")

            if actual.name.date()!=target.date():

                st.info(
                    f"No trading data on {target.date()}. "
                    f"Showing last trading day: "
                    f"{actual.name.date()}."
                )

# ---------- FUTURE ----------
else:

    days=len(pd.bdate_range(
        latest.name.normalize(),target))

    avg_return=d.Close.pct_change().tail(30).mean()

    future_price=latest.Close*(1+avg_return)**days

    future_change=(
        future_price/latest.Close-1)*100

    future_direction=(
        "📈 RISE" if future_change>0 else "📉 FALL")

    a,b,c=st.columns(3)

    with a:
        card("PREDICTED DIRECTION",
             future_direction)

    with b:
        card("ESTIMATED PRICE",
             f"{future_price:,.2f}")

    with c:
        card("EXPECTED CHANGE",
             f"{future_change:.2f}%")

    st.caption(
        "Future values are simulations based on "
        "recent returns, not guaranteed results."
    )

# ==================================================
# STOCK DNA
# ==================================================

st.header("🧬 STOCK DNA")

a,b,c=st.columns(3)

with a:
    card("RSI",f"{latest.RSI:.1f}")

with b:
    card("MACD",f"{latest.MACD:.2f}")

with c:
    card("VOLATILITY",
         f"{latest.VOL*100:.2f}%")

# ==================================================
# TIME MACHINE
# ==================================================

st.header("⏳ TIME MACHINE")

graph(d.index,d.Close,"Price History")
graph(d.index,d.Volume,"Trading Volume")

# ==================================================
# TEST LAB
# ==================================================

st.header("🧪 PREDICTION TEST LAB")

card("HISTORICAL DIRECTION ACCURACY",
     f"{accuracy:.2f}%")

# ==================================================
# TECHNICAL
# ==================================================

st.header("📊 TECHNICAL COMMAND CENTER")

fig,ax=plt.subplots(figsize=(10,3))

ax.plot(d.index,d.Close,label="Price")
ax.plot(d.index,d.MA20,label="MA20")
ax.plot(d.index,d.MA50,label="MA50")

ax.legend()
ax.grid(alpha=.2)

st.pyplot(fig)

graph(d.index,d.RSI,"RSI")
graph(d.index,d.MACD,"MACD")

# ==================================================
# HISTORY
# ==================================================

st.header("📅 RISE / FALL HISTORY")

h=d.tail(10).copy()

h["Change %"]=h.Close.pct_change()*100

h["Direction"]=np.where(
    h["Change %"]>0,
    "📈 RISE",
    "📉 FALL")

st.dataframe(
    h[["Close","Change %","Direction"]],
    use_container_width=True)

# ==================================================
# MARKET RADAR
# ==================================================

st.header("📡 MARKET RADAR")

for n,s in list(STOCKS.items())[:8]:

    q=yf.download(
        s,period="5d",
        auto_adjust=True,
        progress=False)

    if isinstance(q.columns,pd.MultiIndex):
        q.columns=q.columns.get_level_values(0)

    if len(q)>1:

        ch=(q.Close.iloc[-1]/
            q.Close.iloc[-2]-1)*100

        st.write(f"**{n}** → {ch:.2f}%")

# ==================================================
# NEWS
# ==================================================

st.header("📰 NEWS INTELLIGENCE")

try:

    url=f"https://news.google.com/rss/search?q={name}+stock"

    soup=BeautifulSoup(
        requests.get(url,timeout=5).text,
        "xml")

    for n in soup.find_all("item")[:5]:
        st.write("•",n.title.text)

except:

    st.write("News unavailable.")

# ==================================================
# WHAT IF
# ==================================================

st.header("🎮 WHAT-IF MARKET SIMULATOR")

change=st.slider(
    "Market Change %",
    -20.0,20.0,0.0)

whatif=latest.Close*(1+change/100)

card("SIMULATED PRICE",
     f"{whatif:,.2f}")

# ==================================================
# CHALLENGE
# ==================================================

st.header("🏆 PREDICTION CHALLENGE")

if st.button("Test AI Prediction"):

    st.success(
        f"AI predicts {next_direction}")

# ==================================================
# FUTURE SIMULATION
# ==================================================

st.header("🔮 FUTURE SIMULATION")

st.write(
    f"Recent average daily return: "
    f"{d.Close.pct_change().tail(30).mean()*100:.3f}%")

# ==================================================
# VERIFICATION
# ==================================================

st.header("✅ PREDICTION VERIFICATION")

st.write(
    "After the next trading session closes, "
    "compare the AI prediction with the actual "
    "market result using NSE India, Yahoo Finance "
    "or TradingView."
)

# ==================================================
# DATA
# ==================================================

st.header("📋 LATEST DATA")

st.dataframe(
    d.tail(10)[
        ["Close","MA20","MA50",
         "RSI","MACD","Volume"]
    ],
    use_container_width=True)