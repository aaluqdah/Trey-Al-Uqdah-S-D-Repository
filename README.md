# Oil Supply & Demand Forecasting Model

This repository contains a modular Python-based system for forecasting oil market dynamics by modeling **supply and demand (S&D) behavior** across various product streams. The model predicts **consumer and producer behavior** in the current period as a function of price and uses these insights to forecast the **next-period S&D balance**.

---

## Project Structure

The model is split into dedicated Jupyter notebooks for each key component of the petroleum market:
**RUN FILES IN ORDER**

- `Production.ipynb` – U.S. crude production forecast  
- `Gasoline.ipynb` – Gasoline demand
- `Diesel.ipynb` – Diesel (distillate) fuel demand  
- `HeatingOil.ipynb` – Heating oil demand
- `JetFuel.ipynb` – Jet fuel demand 
- `Imports.ipynb` – Crude imports forecast 
- `Exports.ipynb` – Crude exports forecast  
- `Refinery.ipynb` – Refinery Crude demand 
- `S&D.ipynb` – Final model output: Demand (Consumption + Epxports) & Supply (Production + Imports)

---

## Python Environment

- **Python Version:** `3.10.16`
- **Architecture Requirement:** Must use a **64-bit Python environment**

---

## Dependencies

Install dependencies with:

```bash
pip install -r requirements.txt
