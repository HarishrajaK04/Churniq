# ChurnIQ — Customer Churn Prediction

A beginner-friendly Streamlit application for telecom customer analytics and a historical inactivity-proxy classification demo.

## Dataset findings
The provided CSV has 99,999 rows and 226 columns. It has no direct `Churn` label. It includes monthly June–September fields, an `aon` tenure field, and `mobile_number` customer identifiers. The file has no exact duplicate rows and no duplicate `mobile_number` values in the inspected copy.

## Target definition and limitation
The app labels a record as an **inactive proxy** if all available September indicators below are zero or missing:
- outgoing voice minutes (`total_og_mou_9`)
- incoming voice minutes (`total_ic_mou_9`)
- 2G and 3G data volume (`vol_2g_mb_9`, `vol_3g_mb_9`)
- recharge count and amount (`total_rech_num_9`, `total_rech_amt_9`)
- data recharge count and 2G/3G recharge counts, when present

A positive value in any one of these fields marks the customer active/retained for this proxy. This is a **proxy label**, not a confirmed contract cancellation label. In the supplied file this rule produces roughly 2.8% inactive-proxy records; validate this definition before real-world decisions.

## Leakage prevention
The model only uses numeric June–August columns. It excludes all `_9` and `sep_` columns, customer IDs, and date-string fields. Preprocessing is fitted within each pipeline after an 80/20 stratified split. The models are compared using held-out ROC-AUC, F1, precision, recall, and accuracy. The selected fitted pipeline is saved with Joblib.

## Run locally
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place your dataset at `telecom_churn_data.csv` in this folder, or upload it through the sidebar after launch.
5. Start the app:
   ```bash
   streamlit run app.py
   ```
6. Open **Model Lab** and click **Train and evaluate models**. Training may take a few minutes.

## Project structure
- `app.py` — Streamlit UI, data processing, analytics, model training and predictions.
- `requirements.txt` — Python dependencies.
- `telecom_churn_data.csv` — your local dataset (copy it here; not included in the source ZIP to keep the package smaller).
- `artifacts/` — created automatically to store the trained pipeline.

## Interview explanation
- **Problem:** identify customers whose September usage/recharge records indicate inactivity, based on June–August behavior.
- **Features:** numeric historical usage, recharge, ARPU, data, voice, and tenure values from June–August.
- **Target:** multi-signal September inactivity proxy.
- **Evaluation:** stratified held-out test set; compare Logistic Regression and Random Forest using ROC-AUC and F1.
- **Limitations:** the proxy may not represent true churn; class imbalance and missing data matter; threshold bands are illustrative; external validation is required before deployment.
