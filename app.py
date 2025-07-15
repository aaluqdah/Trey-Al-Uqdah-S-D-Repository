import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from dateutil.relativedelta import relativedelta

# === IMPORT ALL SCRIPT FUNCTIONS ===
from Production import main as run_production
from Gasoline import main as run_gasoline
from Diesel import main as run_diesel
from HeatingOil import main as run_heatingoil
from JetFuel import main as run_jetfuel
from Imports import main as run_imports
from Exports import main as run_exports
from Refinery import main as run_refinery
from SD import main as run_sd  # Rename S&D.ipynb to SD.py

# === STREAMLIT CONFIG ===
st.set_page_config(page_title="Oil S&D Dashboard", layout="wide")
st.title("Oil Supply & Demand Forecasting Dashboard")

# === RUN SCRIPTS ===
@st.cache_resource(show_spinner=False)
def run_scripts():
    run_production()
    run_gasoline()
    run_diesel()
    run_heatingoil()
    run_jetfuel()
    run_imports()
    run_exports()
    run_refinery()
    run_sd()
    return True

@st.cache_data(show_spinner=False)
def load_df():
    return pd.read_csv("df_merged2.csv", parse_dates=["ds"])

with st.spinner("Running full model pipeline..."):
    run_scripts()
    df_merged2 = load_df()

if df_merged2 is None:
    st.error("Could not load df_merged2. Make sure it's defined in SD.py.")
    st.stop()

# === FORECAST TABLE ===
st.subheader("Forecast Table")
st.dataframe(df_merged2[["ds", "Supply", "Demand", "Spread"]].tail(12), use_container_width=True)

# === PLOT 1: Spread + Stock Change + WTI ===
st.subheader("Spread and Stock Change vs WTI")
df = df_merged2.copy()
df["ds"] = pd.to_datetime(df["ds"])
df = df[df["ds"] >= "2020-01-01"].dropna(subset=["WTI", "Spread", "Stock Change"])
df["WTI"] = pd.to_numeric(df["WTI"], errors="coerce")

spread_min, spread_max = df["Spread"].min(), df["Spread"].max()
change_min, change_max = df["Stock Change"].min(), df["Stock Change"].max()
spread_zero_pos = abs(spread_min) / (spread_max - spread_min)
change_zero_pos = abs(change_min) / (change_max - change_min)
target_zero_pos = max(spread_zero_pos, change_zero_pos)

spread_total_range = spread_max / (1 - target_zero_pos)
change_total_range = change_max / (1 - target_zero_pos)

spread_ylim = [-spread_total_range * target_zero_pos, spread_total_range * (1 - target_zero_pos)]
change_ylim = [-change_total_range * target_zero_pos, change_total_range * (1 - target_zero_pos)]

start, end = df["ds"].min(), df["ds"].max() + relativedelta(months=1)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

ax1.plot(df["ds"], df["Spread"], color="blue", linewidth=2, label="Spread")
ax1.set_ylabel("Spread", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.set_ylim(spread_ylim)
ax1.set_title("Spread and Monthly Stock Change Since 2020")
ax1.grid(True)

ax1b = ax1.twinx()
bar_colors = df["Stock Change"].apply(lambda x: "green" if x >= 0 else "red")
ax1b.bar(df["ds"], df["Stock Change"], color=bar_colors, width=20, alpha=0.5)
ax1b.set_ylabel("Stock Change", color="gray")
ax1b.tick_params(axis="y", labelcolor="gray")
ax1b.set_ylim(change_ylim)

ax2.plot(df["ds"], df["WTI"], color="orange", linewidth=2, label="WTI Price")
ax2.set_ylabel("Price (USD/barrel)")
ax2.set_title("WTI Crude Oil Price Since 2020")
ax2.grid(True)
ax2.legend()

ax2.set_xlim([start, end])
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=0)
plt.tight_layout()
st.pyplot(fig)

# === PLOT 2: Forecasted Supply and Demand ===
st.subheader("U.S. WTI Crude Supply S&D Balance Forecast")
df = df_merged2.copy()
df['ds'] = pd.to_datetime(df['ds'])
last_date = df['ds'].max()
start_date = last_date - pd.DateOffset(months=23)
df_recent = df[df['ds'] >= start_date].copy()

hist = df_recent.iloc[:-3].copy()
fcast = df_recent.iloc[-3:].copy()

fcast_supply = pd.concat([hist[['ds', 'Supply']].iloc[[-1]], fcast[['ds', 'Supply']]])
fcast_demand = pd.concat([hist[['ds', 'Demand']].iloc[[-1]], fcast[['ds', 'Demand']]])
x_start = df_recent['ds'].min()
x_end = df_recent['ds'].max() + relativedelta(months=1)

fig2, ax = plt.subplots(figsize=(12, 6))
ax.plot(hist['ds'], hist['Supply'], color='red', label='Supply')
ax.plot(hist['ds'], hist['Demand'], color='blue', label='Demand')
ax.plot(fcast_supply['ds'], fcast_supply['Supply'], color='red', linestyle='--', marker='o', label='Forecasted Supply')
ax.plot(fcast_demand['ds'], fcast_demand['Demand'], color='blue', linestyle='--', marker='o', label='Forecasted Demand')
ax.axvline(x=fcast['ds'].iloc[0], color='gray', linestyle=':', label='Forecast Start')

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
ymin, ymax = pd.concat([df_recent['Supply'], df_recent['Demand']]).agg(['min', 'max'])
ax.set_ylim(ymin - 0.5 * (ymax - ymin), ymax + 0.5 * (ymax - ymin))
ax.set_title('U.S. Crude Oil Supply and Demand')
ax.set_xlabel('Date')
ax.set_ylabel('Volume (MBBL/D)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.xlim([x_start, x_end])
st.pyplot(fig2)

# === PLOT 3: Inventory and Stock Change ===
st.subheader("Inventory and Stock Change")
df = df_merged2.copy()
df['ds'] = pd.to_datetime(df['ds'])
end_date = df['ds'].max()
start_date = end_date - relativedelta(months=23)
df = df[(df['ds'] >= start_date) & (df['ds'] <= end_date)]

fig3, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df['ds'], df['Stocks'], color='blue', label='Stocks', linewidth=2.5)
ax1.set_ylabel("Stocks (MMBBL)", color="black")
ax1.tick_params(axis="y", labelcolor="black")
ax1.set_title("Stocks and Monthly Stock Change")
ax1.grid(True)

ax2 = ax1.twinx()
bar_colors = df["Stock Change"].apply(lambda x: 'green' if pd.notna(x) and x >= 0 else 'red')
ax2.bar(df["ds"], df["Stock Change"], color=bar_colors, width=20, alpha=0.5, label="Stock Change")
ax2.set_ylabel("Stock Change (MBBL/D)", color="gray")
ax2.tick_params(axis="y", labelcolor="gray")
ax1.set_xticks(df['ds'])
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
ax1.set_xlim([start_date, end_date + relativedelta(months=0)])
fig3.tight_layout()
st.pyplot(fig3)

