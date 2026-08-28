import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from data import generate_synthetic_data, load_custom_data, load_creditcard_data
from models import train_model_by_strategy, predict_with_threshold
from metrics import calculate_metrics
from utils import (
    get_strategy_description,
    get_model_description,
    get_strategy_story,
    get_model_story,
    get_strategy_hint,
    get_model_hint,
    get_metrics_story,
)

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Cost-Sensitive Learning Educational Tool",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# Global styling: fonts, colors, cards
# ----------------------------------------------------------------------
ACCENT = "#6C5CE7"
ACCENT_SOFT = "#F6F4FE"
GOOD = "#009E73"  # Okabe-Ito bluish green - colorblind-safe
BAD = "#D55E00"   # Okabe-Ito vermillion - colorblind-safe, distinguishable from GOOD

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}
h1, h2, h3, h4 {{
    font-family: 'Poppins', sans-serif !important;
    font-weight: 700 !important;
}}
h1 {{
    background: linear-gradient(90deg, {ACCENT}, #A29BFE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    padding-bottom: 4px;
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {ACCENT_SOFT};
    border-right: 1px solid #E4DEFA;
}}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
    color: {ACCENT};
}}

/* Cards */
.fun-card {{
    background: white;
    border: 1px solid #ECE9FB;
    border-radius: 16px;
    padding: 22px 26px;
    box-shadow: 0 4px 14px rgba(108, 92, 231, 0.07);
    margin-bottom: 14px;
}}
.step-card {{
    background: white;
    border-radius: 14px;
    padding: 16px 18px;
    border: 1px solid #ECE9FB;
    height: 100%;
    box-shadow: 0 2px 8px rgba(108,92,231,0.06);
}}
.step-num {{
    display:inline-block;
    background: {ACCENT};
    color: white;
    font-weight: 700;
    width: 26px; height: 26px;
    border-radius: 50%;
    text-align: center;
    line-height: 26px;
    margin-bottom: 6px;
    font-family: 'Poppins', sans-serif;
}}
.pill {{
    display:inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 6px;
}}
.pill-bad {{ background:#FDEEEA; color:{BAD}; }}
.pill-good {{ background:#E6FAF5; color:{GOOD}; }}
.pill-accent {{ background:{ACCENT_SOFT}; color:{ACCENT}; }}

div[data-testid="stMetric"] {{
    background: white;
    border: 1px solid #ECE9FB;
    border-radius: 14px;
    padding: 14px 10px 6px 14px;
    box-shadow: 0 2px 8px rgba(108,92,231,0.06);
}}
hr {{ border-color: #ECE9FB !important; }}
</style>
""", unsafe_allow_html=True)


def fun_card(content: str):
    st.markdown(f'<div class="fun-card">{content}</div>', unsafe_allow_html=True)


PLOTLY_COLORS = ["#B2BEC3", ACCENT, "#56B4E9", "#FDCB6E"]  # grey, purple, sky blue, yellow - each strategy gets its own distinct, colorblind-safe color, separate from GOOD (used only for the actual winner)

# ----------------------------------------------------------------------
# TOP SECTION: Hero
# ----------------------------------------------------------------------
st.title("🎯 Cost-Sensitive Learning Educational Tool")

fun_card("""
<b>Imagine you're a smoke detector.</b> There are two ways you can mess up: you can stay silent while
there's a real fire (bad, someone could get hurt), or you can blast the alarm when someone's just
making toast (annoying, but nobody dies). Both are "wrong," but they clearly don't cost the same.
<br><br>
That's the whole idea behind this tool. Machine learning models make the exact same kind of mistakes,
and by default, most models treat every mistake as equally bad. This tool lets you tell the model
<b>"actually, missing the real thing is way worse than a false alarm"</b> (or vice versa), and shows you
three different tricks for teaching it that, plus what it actually saves you in the end.
""")

with st.expander("🧭 How to use this tool", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("Pick your data", "Choose synthetic data, real credit card fraud data, or upload your own CSV on the left."),
        ("Set your costs", "Tell the tool how bad a 'missed case' is versus a 'false alarm', in your own numbers."),
        ("Pick a strategy", "Choose one of three ways to teach the model about those costs."),
        ("See the payoff", "Watch the Total Misclassification Cost, the real-world price tag of getting it wrong."),
    ]
    for col, (title, desc) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-num">{steps.index((title, desc)) + 1}</div>
                <b>{title}</b>
                <p style="font-size:0.85rem; color:#555; margin-top:4px;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

st.write("")

# ----------------------------------------------------------------------
# KEY CONCEPT: Data imbalance vs Cost imbalance, in plain English
# ----------------------------------------------------------------------
with st.expander("🤔 New to this? Click here, 'data imbalance' vs 'cost imbalance' "):
    colA, colB = st.columns(2)
    with colA:
        st.markdown("##### 📊 Data Imbalance")
        st.markdown("""
        This just means one outcome is *rare*. Picture a jar of 100 jellybeans where 99 are blue
        and only 1 is red. If you built a machine that always guessed "blue," it would be right
        99% of the time: and still *never once* find the red jellybean.

        That's exactly what happens with things like fraud (rare), disease (rare), or
        equipment failure (rare). A model can look "99% accurate" on paper while being
        completely useless at the one thing you actually care about. This is called the
        **Accuracy Paradox**, and it's the #1 trap beginners fall into.
        """)
    with colB:
        st.markdown("##### ⚖️ Cost Imbalance")
        st.markdown("""
        This is a *different* problem: even if the data were perfectly balanced, the two
        kinds of mistakes might not hurt equally. Missing one real cancer case is a much
        bigger deal than sending someone for one extra, unnecessary scan, even if both
        happen equally often.

        **This tool is all about cost imbalance.** You get to decide, in real numbers,
        how much worse one mistake is than the other, and the tool shows you which
        strategy saves you the most "damage" overall.
        """)

    st.markdown("---")
    st.markdown("""
    **One more twist:** the words "Positive" and "Negative" here don't mean good or bad: they're
    just labels, like heads or tails on a coin. If "Positive" means "has cancer," then:
    - **False Negative** = the model says "healthy" but they actually have cancer *(a missed case)*
    - **False Positive** = the model says "cancer" but they're actually healthy *(a false alarm)*

    Swap the labels and the *names* flip, but the underlying mistakes are still there. There's no
    mistake that's universally worse: it always depends on your situation, which is exactly what
    you get to control with the sliders on the left. 👈
    """)

st.divider()

# ----------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------
st.sidebar.header("🎛️ Controls")

# 1. Data Source
st.sidebar.subheader("1️⃣ Data Source")
data_source = st.sidebar.radio(
    "Choose data source",
    ["Synthetic Data", "Credit Card Fraud (Real)", "Upload Your Own CSV"]
)

if data_source == "Synthetic Data":
    st.sidebar.caption("Fake, made-up data you can shape however you like, great for experimenting freely.")
    imbalance_ratio = st.sidebar.select_slider(
        "Majority class proportion (Class 0)",
        options=[0.50, 0.70, 0.80, 0.90, 0.95, 0.99],
        value=0.50
    )
    class_sep = st.sidebar.slider("Class Separation", 0.5, 2.0, 1.0, 0.1,
                                   help="How easy it is to tell the two groups apart. Higher = easier problem.")
    n_samples = st.sidebar.slider("Number of samples", 500, 3000, 1000, 100)

    X_train, X_test, y_train, y_test = generate_synthetic_data(
        n_samples=n_samples,
        imbalance_ratio=imbalance_ratio,
        class_sep=class_sep
    )
    st.sidebar.success(f"✅ Train: {len(y_train)} | Test: {len(y_test)}")

elif data_source == "Credit Card Fraud (Real)":
    st.sidebar.caption("A real, famous dataset of anonymised bank transactions where fraud is rare.")
    try:
        with st.spinner("Loading Credit Card Fraud data..."):
            X_train, X_test, y_train, y_test = load_creditcard_data(sample_size=30000)
        st.sidebar.success(f"✅ Real data loaded: Train: {len(y_train)} | Test: {len(y_test)}")
    except Exception as e:
        st.sidebar.error(str(e))
        st.stop()

else:
    st.sidebar.caption("Bring your own data. Last column must be 0/1 (the outcome you're predicting).")
    uploaded_file = st.sidebar.file_uploader("Upload CSV (last column = target 0/1)", type=["csv"])
    if uploaded_file is None:
        st.sidebar.warning("Please upload a CSV file.")
        st.stop()
    try:
        X_train, X_test, y_train, y_test, _, _ = load_custom_data(uploaded_file)
        st.sidebar.success(f"✅ Loaded: Train: {len(y_train)} | Test: {len(y_test)}")
    except Exception as e:
        st.sidebar.error(str(e))
        st.stop()

# 2. Cost Settings
st.sidebar.subheader("2️⃣ Cost Settings")

with st.sidebar.expander("💡 What are these costs?", expanded=False):
    st.markdown("""
    A **Missed Target (FN)** = the model says *"nothing to see here"* but it was actually
    the important case. *(A hospital scan says "healthy" but the patient has cancer.)*

    A **False Alarm (FP)** = the model says *"this is it!"* but it wasn't. *(A healthy
    patient gets told they might have cancer, and needs a follow-up test.)*

    Neither is universally worse: you decide, using the numbers below.
    """)

fn_cost = st.sidebar.number_input(
    "💥 Cost of Missing a Target (FN)",
    min_value=1.0,
    value=10.0,
    step=1.0,
    help="How expensive is it when the model fails to catch the important case?"
)
fp_cost = st.sidebar.number_input(
    "🔔 Cost of a False Alarm (FP)",
    min_value=1.0,
    value=1.0,
    step=1.0,
    help="How expensive is it when the model raises a false alarm?"
)

st.sidebar.markdown(f"""
<div class="pill pill-accent">Missing a target costs {fn_cost/fp_cost:.0f}x more than a false alarm</div>
""", unsafe_allow_html=True)

# 3. Model
st.sidebar.subheader("3️⃣ Model")
model_choice = st.sidebar.selectbox(
    "Base classifier",
    ["Logistic Regression", "Decision Tree", "Random Forest", "SVM"]
)
st.sidebar.caption(get_model_hint(model_choice))

# 4. Strategy
st.sidebar.subheader("4️⃣ Strategy")
strategy = st.sidebar.selectbox(
    "Cost-sensitive strategy",
    ["Threshold Moving", "Class Weighting", "Resampling (SMOTE)"]
)
st.sidebar.caption(get_strategy_hint(strategy))

# ----------------------------------------------------------------------
# Train models (including Cost-Blind Baseline)
# ----------------------------------------------------------------------
strategies = ["Threshold Moving", "Class Weighting", "Resampling (SMOTE)"]
results = {}

baseline_model, _ = train_model_by_strategy(
    X_train, y_train,
    strategy="Threshold Moving",
    model_type=model_choice,
    fn_cost=1.0,
    fp_cost=1.0
)
baseline_pred = predict_with_threshold(baseline_model, X_test, threshold=0.5)
baseline_metrics = calculate_metrics(y_test, baseline_pred, fn_cost=fn_cost, fp_cost=fp_cost)
results["Cost-Blind (0.5)"] = {
    "model": baseline_model,
    "threshold": 0.5,
    "y_pred": baseline_pred,
    "metrics": baseline_metrics
}

for strat in strategies:
    model, thresh = train_model_by_strategy(
        X_train, y_train,
        strategy=strat,
        model_type=model_choice,
        fn_cost=fn_cost,
        fp_cost=fp_cost
    )
    y_pred = predict_with_threshold(model, X_test, threshold=thresh)
    metrics = calculate_metrics(y_test, y_pred, fn_cost=fn_cost, fp_cost=fp_cost)

    results[strat] = {
        "model": model,
        "threshold": thresh,
        "y_pred": y_pred,
        "metrics": metrics
    }

current = results[strategy]
current_metrics = current["metrics"]

# ----------------------------------------------------------------------
# HERO METRIC: Total Misclassification Cost
# ----------------------------------------------------------------------
st.markdown("## 💰 Total Misclassification Cost")
fun_card("""
This is the single number to watch. Think of it as <b>the total bill</b> for every mistake your
model made, using the price tags you set on the left. <b>Lower is always better</b>, £0 would mean
a perfect model.
""")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label=f"💷 Total Cost, {strategy}",
        value=f"£{current_metrics['Total Cost']:,.0f}"
    )
with col2:
    st.metric("❌ Missed Targets (FN)", current_metrics["FN"],
              help="Model said No but answer was Yes")
with col3:
    st.metric("🚨 False Alarms (FP)", current_metrics["FP"],
              help="Model said Yes but answer was No")

st.caption(
    f"How this is worked out: "
    f"{current_metrics['FN']} missed targets × £{fn_cost:.0f} "
    f"+ {current_metrics['FP']} false alarms × £{fp_cost:.0f} "
    f"= **£{current_metrics['Total Cost']:,.0f}**"
)

st.markdown(f"**Decision threshold used:** `{current['threshold']:.3f}`")

if strategy == "Threshold Moving":
    st.info(
        f"📐 Elkan's formula: threshold = FP cost ÷ (FP cost + FN cost) "
        f"= {fp_cost} ÷ ({fp_cost} + {fn_cost}) = **{fp_cost/(fp_cost+fn_cost):.3f}**"
    )

# Real world domain examples: as tabs, more fun than a wall of bullets
st.markdown("#### 🌍 Where does this show up in real life?")
tab_med, tab_fraud, tab_ads = st.tabs(["🏥 Medical Screening", "💳 Credit Card Fraud", "📢 Advertising"])
with tab_med:
    st.markdown("""
    - **Missed case (FN):** patient has a disease, model says healthy → treatment delayed, could be life-threatening
    - **False alarm (FP):** healthy patient flagged → extra tests, stressful but manageable
    - **Typical cost ratio:** missing a case is *far* worse than a false alarm
    """)
with tab_fraud:
    st.markdown("""
    - **Missed fraud (FN):** a fraudulent transaction goes through → the bank loses the money
    - **False alarm (FP):** a genuine transaction gets blocked → annoyed customer, possible churn
    - **Typical cost ratio:** depends on transaction size and how valuable the customer is
    """)
with tab_ads:
    st.markdown("""
    - **Missed opportunity (FN):** an interested customer never gets shown the ad → lost sale
    - **False alarm (FP):** an uninterested customer gets shown the ad → a bit of wasted ad spend
    - **Typical cost ratio:** usually much more balanced than medicine or fraud
    """)
st.caption("The lesson across all three: there's no mistake that's universally worse. Your numbers define what matters.")

st.divider()

# ----------------------------------------------------------------------
# Before vs After: interactive reveal
# ----------------------------------------------------------------------
st.subheader("🎬 Before vs After: Does Teaching Cost Actually Pay Off?")
fun_card(f"""
Here's the moment of truth. The <b>Before</b> card is your cost-blind baseline model, it just uses a
plain 50/50 cutoff and has never heard about your cost settings. Hit the button to reveal what happens
when it's swapped for the <b>{strategy}</b> strategy you picked on the left, tuned to your exact numbers.
""")

if "reveal_strategy" not in st.session_state:
    st.session_state.reveal_strategy = False

baseline_cost = results["Cost-Blind (0.5)"]["metrics"]["Total Cost"]
strategy_cost = current_metrics["Total Cost"]
savings = baseline_cost - strategy_cost
pct_savings = (savings / baseline_cost * 100) if baseline_cost > 0 else 0.0

just_revealed = False
btn_label = "🔄 Hide the result" if st.session_state.reveal_strategy else "🔍 Reveal what cost-aware learning does"
if st.button(btn_label, use_container_width=True):
    st.session_state.reveal_strategy = not st.session_state.reveal_strategy
    just_revealed = st.session_state.reveal_strategy

col_before, col_after = st.columns(2)
with col_before:
    st.markdown("##### 😐 Before, Cost-Blind Model")
    st.metric("Total Cost", f"£{baseline_cost:,.0f}")
    st.caption("Plain 50/50 cutoff. Has never heard about your cost settings.")

with col_after:
    if st.session_state.reveal_strategy:
        st.markdown(f"##### 🎉 After, {strategy}")
        st.metric(
            "Total Cost",
            f"£{strategy_cost:,.0f}",
            delta=f"{'-' if savings >= 0 else '+'}£{abs(savings):,.0f} ({pct_savings:+.1f}%)",
            delta_color="normal" if savings >= 0 else "inverse",
        )
        st.caption(f"Uses {strategy}, tuned to your exact cost settings.")
    else:
        st.markdown("##### ❔ After, ???")
        st.metric("Total Cost", "£ ? ? ?")
        st.caption("Click the button above to find out!")

if st.session_state.reveal_strategy:
    if savings > 0:
        st.success(f"✅ Teaching the model about costs saved you **£{savings:,.0f}** ({pct_savings:.1f}% cheaper)!")
        if just_revealed:
            st.balloons()
    elif savings < 0:
        st.warning(
            f"⚠️ In this case the cost-blind model was actually £{abs(savings):,.0f} cheaper. "
            f"Try a different strategy or model on the left: not every combination wins!"
        )
    else:
        st.info("Same cost either way here, try nudging your FN/FP cost sliders to see this shift.")

st.divider()

# ----------------------------------------------------------------------
# Cost Comparison across all strategies: interactive Plotly
# ----------------------------------------------------------------------
st.subheader("📊 Cost Comparison Across All Strategies")
fun_card("""
This chart is the payoff of the whole tool. The <b>grey bar</b> is a model that has never been told
about your costs at all, it just uses the "obvious" 50/50 cutoff. The colourful bars are the three
cost-aware strategies. Hover over each bar to see the exact numbers.
""")

order = ["Cost-Blind (0.5)", "Threshold Moving", "Class Weighting", "Resampling (SMOTE)"]
cost_values = [results[name]["metrics"]["Total Cost"] for name in order]
best_strat = min(order, key=lambda x: results[x]["metrics"]["Total Cost"])

# Worst-case reference: the costlier of the two trivial "always guess the same answer" policies.
# This is the exact Accuracy Paradox scenario from the top of the page, now with a real price tag.
n_actual_pos = int(np.sum(y_test == 1))
n_actual_neg = int(np.sum(y_test == 0))
worst_always_negative = n_actual_pos * fn_cost   # never flag anything: every real positive becomes a costly FN
worst_always_positive = n_actual_neg * fp_cost   # flag everything: every real negative becomes a FP
worst_case_cost = max(worst_always_negative, worst_always_positive)
worst_case_label = "always saying 'no'" if worst_always_negative >= worst_always_positive else "always saying 'yes'"

bar_colors = [
    GOOD if name == best_strat else (PLOTLY_COLORS[i] if name != "Cost-Blind (0.5)" else "#B2BEC3")
    for i, name in enumerate(order)
]

fig = go.Figure(data=[
    go.Bar(
        x=order,
        y=cost_values,
        marker_color=bar_colors,
        text=[f"£{v:,.0f}" for v in cost_values],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Total Cost: £%{y:,.0f}<extra></extra>",
    )
])
fig.add_hline(
    y=worst_case_cost,
    line_dash="dash",
    line_color=BAD,
    annotation_text=f"Worst case ({worst_case_label} every time): £{worst_case_cost:,.0f}",
    annotation_position="top left",
    annotation_font_color=BAD,
)
fig.update_layout(
    title="Which strategy gives the lowest total cost? (Lower is better)",
    yaxis_title="Total Misclassification Cost (£)",
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Inter, sans-serif", size=13),
    margin=dict(t=60, b=10),
    height=420,
)
fig.update_yaxes(gridcolor="#ECE9FB")
st.plotly_chart(fig, use_container_width=True)

st.success(f"🏆 Lowest cost strategy under your current settings: **{best_strat}**")

st.markdown("""
**What this chart teaches you:**
- The grey bar ignores your costs completely, it's the "before" picture
- The coloured bars each use your cost settings, just applied at a different stage
- The dashed red line is the worst case, giving the same answer every time no matter what. That's the Accuracy Paradox from earlier, now with a real price tag
- Nudge the FN and FP cost sliders on the left and watch the winner change, that's the whole point!
""")

st.divider()

# ----------------------------------------------------------------------
# Confusion Matrix & Performance Metrics: as tabs
# ----------------------------------------------------------------------
st.subheader("🔎 Look Under the Hood")
fun_card("""
The Total Cost number above is the headline, but you don't have to just take it on faith. This section
shows the actual numbers it's built from, so you can check the maths yourself, and see the classic
metrics data scientists use, to compare against.
""")
tab_cm, tab_perf = st.tabs(["🧩 Confusion Matrix", "📈 Performance Metrics"])

with tab_cm:
    st.markdown(f"#### Confusion Matrix, {strategy}")
    fun_card("""
    A confusion matrix is just a <b>report card</b> that sorts every prediction into one of four boxes:
    got it right (twice), or got it wrong (in one of two ways). No maths needed: just read the grid.
    """)

    cm = current_metrics["Confusion Matrix"]
    labels = ["Class 0 (Negative)", "Class 1 (Positive)"]

    z = cm
    fig_cm = px.imshow(
        z,
        text_auto=True,
        color_continuous_scale=["#F6F4FE", ACCENT],
        labels=dict(x="Predicted", y="Actually", color="Count"),
        x=labels,
        y=labels,
    )
    fig_cm.update_traces(textfont_size=18, textfont_color="white")
    fig_cm.update_layout(
        height=420,
        font=dict(family="Inter, sans-serif", size=13),
        margin=dict(t=20),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown(f"""
    | | Predicted Negative | Predicted Positive |
    |---|---|---|
    | **Actually Negative** | ✅ TN = {current_metrics['TN']} (correct) | 🚨 FP = {current_metrics['FP']} (false alarm) |
    | **Actually Positive** | ❌ FN = {current_metrics['FN']} (missed) | ✅ TP = {current_metrics['TP']} (correct) |
    """)

    tp, fp, fn = current_metrics["TP"], current_metrics["FP"], current_metrics["FN"]
    st.caption(
        f"**Precision** = TP ÷ (TP + FP) = {tp} ÷ ({tp} + {fp}) = **{current_metrics['Precision']:.0%}**"
    )
    st.caption(
        f"**Recall** = TP ÷ (TP + FN) = {tp} ÷ ({tp} + {fn}) = **{current_metrics['Recall']:.0%}**"
    )
    st.caption("These numbers come straight from the table above, feel free to check them on a calculator.")

    st.markdown("""
    <span class="pill pill-good">TP / TN = correct</span>
    <span class="pill pill-bad">FP / FN = mistakes</span>
    """, unsafe_allow_html=True)

with tab_perf:
    st.markdown("#### Performance Metrics Comparison")
    fun_card("""
    These are the classic "report card" scores data scientists use. Notice how <b>accuracy can look
    great even when the Total Cost is high</b>, that's the Accuracy Paradox from earlier, showing up
    in real numbers.
    """)

    with st.expander("💡 What do these metrics actually mean?", expanded=False):
        st.markdown("""
        **Precision** — of everything the model flagged as positive, how much was actually correct?
        A model with low precision cries wolf a lot.

        **Recall** — of everything that was actually positive, how much did the model actually catch?
        A model with low recall misses a lot of real cases.

        **Accuracy** — the fraction of all predictions that were correct. Looks great on paper, but can
        hide a genuinely bad model when the data is imbalanced, that's the Accuracy Paradox.

        **Balanced Accuracy** — averages the accuracy of each class separately, instead of lumping
        everything together. A model that's perfect on the easy majority class and terrible on the rare
        one gets caught immediately here, instead of hidden behind one flattering number.

        **F1-Score** — one single number that balances Precision and Recall together, handy for a quick
        overall sense without having to weigh the two separately.
        """)

    with st.expander("📖 Tell me a story instead"):
        st.markdown(get_metrics_story())

    perf_order = ["Cost-Blind (0.5)", "Threshold Moving", "Class Weighting", "Resampling (SMOTE)"]
    perf_data = {
        name: {
            "Accuracy": results[name]["metrics"]["Accuracy"],
            "Balanced Accuracy": results[name]["metrics"]["Balanced Accuracy"],
            "Precision": results[name]["metrics"]["Precision"],
            "Recall": results[name]["metrics"]["Recall"],
            "F1-Score": results[name]["metrics"]["F1-Score"],
        }
        for name in perf_order
    }
    perf_df = pd.DataFrame(perf_data).T

    fig_perf = go.Figure()
    metric_colors = {"Accuracy": "#B2BEC3", "Balanced Accuracy": ACCENT,
                      "Precision": GOOD, "Recall": "#FDCB6E", "F1-Score": BAD}
    for col in perf_df.columns:
        fig_perf.add_trace(go.Bar(
            name=col,
            x=perf_df.index,
            y=perf_df[col],
            marker_color=metric_colors.get(col),
            hovertemplate="<b>%{x}</b><br>" + col + ": %{y:.3f}<extra></extra>",
        ))
    fig_perf.update_layout(
        barmode="group",
        yaxis_title="Score (0 to 1)",
        yaxis_range=[0, 1.05],
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        height=460,
        margin=dict(t=20),
    )
    fig_perf.update_yaxes(gridcolor="#ECE9FB")
    st.plotly_chart(fig_perf, use_container_width=True)

    st.dataframe(perf_df.style.format("{:.3f}").background_gradient(cmap="Purples", axis=0),
                 use_container_width=True)

    st.markdown("""
    **What to notice:**
    - High accuracy does **not** mean low cost
    - Recall (how many real cases got caught) often matters more than accuracy
    - Balanced Accuracy is fairer than plain Accuracy when the data is imbalanced
    """)

st.divider()

# ----------------------------------------------------------------------
# Strategy explanations
# ----------------------------------------------------------------------
st.subheader("🧠 Understanding the Selected Strategy")
fun_card(get_strategy_description(strategy))

with st.expander("📖 Tell me a story instead"):
    st.markdown(get_strategy_story(strategy))

with st.expander("⚖️ Compare all three strategies side by side"):
    st.markdown("""
    | Strategy | When it happens | How it works | Plain English |
    |---|---|---|---|
    | 🎯 **Threshold Moving** | After training | Adjusts the decision boundary using your cost ratio | "Same brain, but a more cautious trigger finger" |
    | ⚖️ **Class Weighting** | During training | Tells the model some mistakes are more expensive | "Teach it to care more about the rare, costly case from day one" |
    | 🧬 **Resampling (SMOTE)** | Before training | Creates synthetic minority examples to balance the data | "Show it many more (fake but realistic) examples of the rare case" |

    All three strategies plug in the *same* cost ratio you set on the left: they just apply
    it at a different point in the process. None is universally "best"; that's exactly what
    this tool lets you discover for your own numbers.
    """)

st.subheader(f"🤖 Model used: {model_choice}")
fun_card(get_model_description(model_choice))

with st.expander("📖 Tell me a story instead"):
    st.markdown(get_model_story(model_choice))

with st.expander("🧩 Why these four models were chosen"):
    st.markdown("""
    - **Logistic Regression** 📈, Simple, fast, gives clear probabilities. Best starting point.
    - **Decision Tree** 🌳, Easy to understand and visualise. Shows clear decision rules.
    - **Random Forest** 🌲🌲🌲, Stronger than a single tree. More accurate on complex data.
    - **SVM** ✂️, Can learn complex, curved boundaries. Useful when the classes aren't easy to separate.
    """)

st.divider()
st.caption("✨ All results are calculated on a held-out test set. Tool built for MSc dissertation, University of Liverpool.")
