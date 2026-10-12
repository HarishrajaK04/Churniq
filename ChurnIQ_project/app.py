import os
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")
st.set_page_config(page_title="ChurnIQ | Retention Intelligence", page_icon="📡", layout="wide")

DATA_DEFAULT = "telecom_churn_data.csv"
MODEL_PATH = Path("artifacts/churniq_pipeline.joblib")
SEPTEMBER_ACTIVITY = [
    "total_og_mou_9", "total_ic_mou_9", "vol_2g_mb_9", "vol_3g_mb_9",
    "total_rech_num_9", "total_rech_amt_9", "total_rech_data_9",
    "count_rech_2g_9", "count_rech_3g_9"
]
MONTHS = {6: "June", 7: "July", 8: "August", 9: "September"}

# ---------- Styling ----------
st.markdown("""
<style>
.stApp {background: #0b1120; color: #e5edf8;}
[data-testid="stSidebar"] {background: #101a2d; border-right: 1px solid #26364f;}
[data-testid="stMetric"] {background: #111d31; border: 1px solid #26364f; padding: 18px 18px; border-radius: 14px;}
[data-testid="stMetricLabel"] {color: #9fb0c9;}
[data-testid="stMetricValue"] {color: #f1f7ff;}
h1,h2,h3 {color: #f1f7ff;}
div[data-testid="stPlotlyChart"] {background: #111d31; border: 1px solid #26364f; border-radius: 14px; padding: 8px;}
.small-note {color:#9fb0c9;font-size:0.9rem}
.hero {padding: 12px 0 18px 0;}
</style>
""", unsafe_allow_html=True)

def build_target(df):
    """Proxy label: September has no positive activity across any available core usage/recharge indicator."""
    cols = [c for c in SEPTEMBER_ACTIVITY if c in df.columns]
    if len(cols) < 4:
        raise ValueError("Not enough September activity columns to define the proxy target.")
    activity = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).gt(0).any(axis=1)
    # Keep rows with at least some observed September outcome data; in this dataset the core indicators are populated.
    observed = df[cols].notna().any(axis=1)
    out = df.loc[observed].copy()
    out["churn_proxy"] = (~activity.loc[observed]).astype(int)
    return out, cols

def model_features(df):
    # Only June-August predictors. Remove IDs and date strings. This deliberately excludes all September columns.
    drop = {"mobile_number", "churn_proxy"}
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        if c.endswith("_9") or c.startswith("sep_") or "month_9" in c:
            continue
        if c.startswith("date_of_last_rech") or c.startswith("last_date_of_month"):
            continue
        if c == "circle_id":
            # Nominal identifier-like field; exclude rather than learn geography codes as a false signal.
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    # Drop all-null and constant columns
    cols = [c for c in cols if df[c].notna().any() and df[c].nunique(dropna=True) > 1]
    return cols

def train_evaluate(df):
    labeled, target_cols = build_target(df)
    features = model_features(labeled)
    if len(features) < 5:
        raise ValueError("Too few usable June-August numeric features.")
    X = labeled[features].replace([np.inf, -np.inf], np.nan)
    y = labeled["churn_proxy"]
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("The derived proxy target does not contain enough examples of both classes.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), features)
    ], remainder="drop")
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        )
    }
    results, pipelines = {}, {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocess", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "ROC-AUC": float(roc_auc_score(y_test, prob)),
            "F1": float(f1_score(y_test, pred, zero_division=0)),
            "Precision": float(precision_score(y_test, pred, zero_division=0)),
            "Recall": float(recall_score(y_test, pred, zero_division=0)),
            "Accuracy": float(accuracy_score(y_test, pred)),
            "y_test": y_test, "pred": pred, "prob": prob,
            "confusion": confusion_matrix(y_test, pred)
        }
        pipelines[name] = pipe
    # Pick by ROC-AUC, with F1 as tie-breaker.
    best_name = max(results, key=lambda n: (results[n]["ROC-AUC"], results[n]["F1"]))
    artifact = {"pipeline": pipelines[best_name], "model_name": best_name,
                "features": features, "metrics": {n: {k:v for k,v in m.items() if k not in
                ("y_test","pred","prob","confusion")} for n,m in results.items()},
                "target_columns": target_cols, "target_rule": "September core activity indicators all zero or missing"}
    return labeled, features, results, artifact

@st.cache_data(show_spinner=False)
def load_csv(file_bytes, filename):
    import io
    return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)

def money(v):
    return f"₹{v:,.0f}"

def section_title(title, subtitle=None):
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"<div class='small-note'>{subtitle}</div>", unsafe_allow_html=True)

st.sidebar.markdown("# 📡 ChurnIQ")
st.sidebar.caption("Telecom Customer Retention Intelligence")
st.sidebar.markdown("---")
page = st.sidebar.radio("NAVIGATION", ["Overview", "Customer Analytics", "Churn Prediction", "Model Lab", "Data Quality"], label_visibility="visible")
st.sidebar.markdown("---")
uploaded = st.sidebar.file_uploader("Upload telecom CSV", type=["csv"])
data_path = Path(DATA_DEFAULT)
if uploaded is not None:
    raw = load_csv(uploaded.getvalue(), uploaded.name)
elif data_path.exists():
    raw = pd.read_csv(data_path, low_memory=False)
else:
    raw = None
    st.sidebar.warning("Upload your CSV to begin.")

if raw is None:
    st.title("ChurnIQ")
    st.subheader("Telecom Customer Retention Intelligence")
    st.info("Upload the telecom CSV from the sidebar. ChurnIQ will inspect the schema, derive a documented September inactivity proxy label, and train/evaluate models.")
    st.stop()

try:
    labeled, target_cols = build_target(raw)
except Exception as e:
    st.error(f"Unable to construct the historical proxy label: {e}")
    st.stop()

if "model_bundle" not in st.session_state:
    st.session_state.model_bundle = None
if "training_results" not in st.session_state:
    st.session_state.training_results = None

# Dataset-derived metrics
y = labeled["churn_proxy"]
n_customers = len(labeled)
churners = int(y.sum())
retained = int((y == 0).sum())
churn_rate = float(y.mean())
if page == "Overview":
    st.markdown("<div class='hero'>", unsafe_allow_html=True)
    st.title("ChurnIQ")
    st.markdown("### Telecom Customer Retention Intelligence")
    st.markdown("<div class='small-note'>Understand customer behavior, identify inactivity risk, and explore model performance.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if st.session_state.model_bundle:
        st.success(f"Model trained successfully · {st.session_state.model_bundle['model_name']}")
    else:
        st.warning("Model not trained in this session yet. Open Model Lab and train/evaluate the baseline models.")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Customers analyzed", f"{n_customers:,}")
    c2.metric("Inactivity proxy", f"{churners:,}", f"{churn_rate:.2%}")
    c3.metric("Retained / active", f"{retained:,}", f"{1-churn_rate:.2%}")
    if st.session_state.model_bundle:
        best = st.session_state.model_bundle["metrics"][st.session_state.model_bundle["model_name"]]
        c4.metric("Test ROC-AUC", f"{best['ROC-AUC']:.3f}", "Held-out test set")
    else:
        c4.metric("Test ROC-AUC", "—", "Train model first")
    left,right = st.columns([1,1])
    with left:
        section_title("September outcome split", "Proxy label from multiple activity and recharge indicators.")
        fig = px.pie(names=["Inactive proxy", "Active / retained"], values=[churners, retained],
                     hole=.68, color_discrete_sequence=["#22d3ee","#3b82f6"])
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          legend_title="", margin=dict(l=10,r=10,t=15,b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        section_title("Recharge and usage trends", "Monthly means calculated from available June–September columns.")
        trend = []
        for m, label in MONTHS.items():
            for metric, pretty in [("total_rech_amt", "Recharge amount"), ("total_og_mou", "Outgoing minutes"),
                                   ("total_ic_mou", "Incoming minutes"), ("vol_2g_mb", "2G data (MB)"),
                                   ("vol_3g_mb", "3G data (MB)")]:
                col = f"{metric}_{m}"
                if col in raw:
                    trend.append({"Month":label, "Metric":pretty, "Mean":pd.to_numeric(raw[col],errors="coerce").mean()})
        if trend:
            fig = px.line(pd.DataFrame(trend), x="Month", y="Mean", color="Metric", markers=True)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=10,r=10,t=15,b=10), legend_title="")
            st.plotly_chart(fig, use_container_width=True)
    section_title("How ChurnIQ defines churn", "This dataset has no observed Churn column.")
    st.markdown("""**Historical inactivity proxy:** a customer is labelled inactive when every available September core activity indicator is zero or missing across outgoing calls, incoming calls, 2G/3G data usage, and recharge counts/amounts. Any positive indicator means active/retained for this proxy label.

This is **not verified contractual churn**. A customer may have a temporary pause, use an untracked service, or have incomplete records. Treat results as a portfolio demonstration and validate the definition with a telecom domain expert before business use.""")
    st.caption("Predictive inputs are restricted to numeric June–August features; September columns, dates, and customer identifiers are excluded.")

elif page == "Customer Analytics":
    section_title("Customer Analytics", "Explore the actual records in the uploaded dataset.")
    view = labeled.copy()
    col1,col2,col3 = st.columns(3)
    segment = col1.selectbox("Outcome segment", ["All customers","Inactive proxy","Active / retained"])
    min_tenure, max_tenure = int(pd.to_numeric(view["aon"],errors="coerce").min()), int(pd.to_numeric(view["aon"],errors="coerce").max())
    tenure = col2.slider("Tenure (days)", min_tenure, max_tenure, (min_tenure,max_tenure))
    month = col3.selectbox("Usage month", ["June","July","August","September"])
    if segment == "Inactive proxy": view = view[view.churn_proxy == 1]
    elif segment == "Active / retained": view = view[view.churn_proxy == 0]
    view = view[pd.to_numeric(view["aon"],errors="coerce").between(tenure[0],tenure[1])]
    st.caption(f"{len(view):,} customers match the current filters.")
    a,b = st.columns(2)
    m = {"June":6,"July":7,"August":8,"September":9}[month]
    with a:
        col = f"total_rech_amt_{m}"
        if col in view:
            fig=px.histogram(view, x=col, nbins=45, title=f"Recharge amount · {month}", marginal="box")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
    with b:
        col=f"arpu_{m}"
        if col in view:
            fig=px.histogram(view, x=col, nbins=45, title=f"ARPU · {month}", marginal="box")
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
    a,b=st.columns(2)
    with a:
        cols=[c for c in [f"total_og_mou_{m}",f"total_ic_mou_{m}"] if c in view]
        if cols:
            melted=view[cols].melt(var_name="Call type",value_name="Minutes")
            fig=px.box(melted,x="Call type",y="Minutes",title=f"Voice usage · {month}", points=False)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
    with b:
        cols=[c for c in [f"vol_2g_mb_{m}",f"vol_3g_mb_{m}"] if c in view]
        if cols:
            melted=view[cols].melt(var_name="Data type",value_name="MB")
            fig=px.box(melted,x="Data type",y="MB",title=f"Data usage · {month}", points=False)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
    if "aon" in view:
        fig=px.histogram(view,x="aon",color="churn_proxy",nbins=40,barmode="overlay",
                         title="Tenure distribution by inactivity proxy",
                         labels={"aon":"Age on network (days)","churn_proxy":"Inactive proxy"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig,use_container_width=True)
    corr_candidates=[c for c in ["arpu_6","arpu_7","arpu_8","total_rech_amt_6","total_rech_amt_7","total_rech_amt_8",
                                 "total_og_mou_6","total_og_mou_7","total_og_mou_8","total_ic_mou_6","total_ic_mou_7","total_ic_mou_8",
                                 "vol_2g_mb_6","vol_2g_mb_7","vol_2g_mb_8","vol_3g_mb_6","vol_3g_mb_7","vol_3g_mb_8","aon","churn_proxy"] if c in view]
    if len(corr_candidates)>2:
        corr=view[corr_candidates].corr(numeric_only=True)
        fig=px.imshow(corr,text_auto=".2f",aspect="auto",color_continuous_scale="RdBu_r",zmin=-1,zmax=1,title="Selected feature correlations")
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig,use_container_width=True)

elif page == "Churn Prediction":
    section_title("Churn Prediction", "Score a customer using June–August feature values only.")
    bundle=st.session_state.model_bundle
    if not bundle:
        st.warning("Train the model in Model Lab first.")
        st.stop()
    features=bundle["features"]
    st.info("For a single-customer demo, upload a CSV containing the same June–August feature columns, or select an existing row below. Customer identifiers and September fields are ignored.")
    row_mode=st.radio("Prediction input",["Choose a record from dataset","Upload customer feature CSV"],horizontal=True)
    if row_mode=="Choose a record from dataset":
        id_col="mobile_number" if "mobile_number" in raw else None
        options=list(range(len(raw)))
        idx=st.number_input("Dataset row index",min_value=0,max_value=max(0,len(raw)-1),value=0,step=1)
        row=raw.iloc[[int(idx)]].copy()
        if id_col: st.caption(f"Record selected: {row.iloc[0][id_col]}")
        X=row.reindex(columns=features).replace([np.inf,-np.inf],np.nan)
        try:
            prob=float(bundle["pipeline"].predict_proba(X)[0,1])
            st.metric("Predicted inactivity probability",f"{prob:.1%}")
            risk="High" if prob>=0.60 else ("Medium" if prob>=0.30 else "Low")
            color={"High":"#f87171","Medium":"#fbbf24","Low":"#34d399"}[risk]
            st.markdown(f"<h3 style='color:{color}'>Risk category: {risk}</h3>",unsafe_allow_html=True)
            st.progress(int(round(prob*100)))
            st.caption("Risk thresholds are illustrative dashboard bands, not calibrated operational policy.")
        except Exception as e: st.error(f"Prediction failed: {e}")
    else:
        f=st.file_uploader("Upload feature CSV",type=["csv"],key="prediction_csv")
        if f:
            batch=pd.read_csv(f,low_memory=False)
            missing=[c for c in features if c not in batch.columns]
            if missing:
                st.error(f"Missing {len(missing)} required feature columns. Example: {', '.join(missing[:8])}")
            else:
                X=batch.reindex(columns=features).replace([np.inf,-np.inf],np.nan)
                probs=bundle["pipeline"].predict_proba(X)[:,1]
                out=pd.DataFrame({"predicted_inactivity_probability":probs,
                                  "risk_category":np.where(probs>=.60,"High",np.where(probs>=.30,"Medium","Low"))})
                st.dataframe(out.head(100),use_container_width=True)
                st.download_button("Download predictions CSV",out.to_csv(index=False).encode("utf-8"),"churniq_predictions.csv","text/csv")

elif page == "Model Lab":
    section_title("Model Lab", "Reproducible training and held-out comparison.")
    st.markdown("The model uses an 80/20 stratified train/test split. Imputation and scaling are fitted inside the training pipeline. The final pipeline is selected by test ROC-AUC, with F1 as a tie-breaker.")
    if st.button("Train and evaluate models",type="primary"):
        with st.spinner("Training Logistic Regression and Random Forest. This may take a few minutes..."):
            try:
                labeled2,features,results,artifact=train_evaluate(raw)
                st.session_state.training_results=results
                st.session_state.model_bundle=artifact
                MODEL_PATH.parent.mkdir(parents=True,exist_ok=True)
                joblib.dump(artifact,MODEL_PATH)
                st.success(f"Training complete. Selected: {artifact['model_name']}")
            except Exception as e:
                st.error(f"Training failed: {e}")
    if st.session_state.training_results:
        results=st.session_state.training_results
        rows=[]
        for name,metrics in results.items():
            rows.append({"Model":name,**{k:metrics[k] for k in ["ROC-AUC","F1","Precision","Recall","Accuracy"]}})
        st.dataframe(pd.DataFrame(rows).set_index("Model").style.format("{:.3f}"),use_container_width=True)
        best=st.session_state.model_bundle["model_name"]
        st.markdown(f"**Selected model:** {best}")
        a,b=st.columns(2)
        with a:
            metric_df=pd.DataFrame(rows).melt(id_vars="Model",var_name="Metric",value_name="Score")
            fig=px.bar(metric_df,x="Metric",y="Score",color="Model",barmode="group",title="Held-out test metrics",range_y=[0,1])
            fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
        with b:
            cm=results[best]["confusion"]
            fig=px.imshow(cm,text_auto=True,x=["Predicted active","Predicted inactive"],y=["Actual active","Actual inactive"],
                          labels=dict(x="Prediction",y="Actual",color="Count"),title="Confusion matrix · selected model")
            fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig,use_container_width=True)
        st.caption(f"Model pipeline saved to {MODEL_PATH}. Metrics shown are from the held-out test set, not training accuracy.")
    else:
        st.info("No model has been trained in this session. Click the button above to start.")

elif page == "Data Quality":
    section_title("Data Quality", "Inspection of the actual file loaded by ChurnIQ.")
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Rows",f"{len(raw):,}")
    c2.metric("Columns",f"{raw.shape[1]:,}")
    c3.metric("Exact duplicate rows",f"{int(raw.duplicated().sum()):,}")
    c4.metric("Duplicate customer IDs",f"{int(raw['mobile_number'].duplicated().sum()):,}" if "mobile_number" in raw else "N/A")
    st.markdown("### Column types")
    st.dataframe(raw.dtypes.astype(str).value_counts().rename_axis("Data type").to_frame("Columns"),use_container_width=True)
    st.markdown("### Highest missing-value rates")
    missing=(raw.isna().mean()*100).sort_values(ascending=False).head(25).rename("Missing %").reset_index(names="Column")
    st.dataframe(missing.style.format({"Missing %":"{:.1f}%"}),use_container_width=True)
    st.markdown("### Target rule inputs")
    st.write(target_cols)
    st.markdown("### Sample records")
    st.dataframe(raw.head(10),use_container_width=True)
    st.download_button("Download data-quality summary",missing.to_csv(index=False).encode("utf-8"),"churniq_missingness.csv","text/csv")
