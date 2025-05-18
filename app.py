import streamlit as st
import os
import pickle
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy import stats

# === Feature name mapping ===
feature_name_map = {
    'Sa': 'Sa (g)',
    'No.story': 'No. Stories',
    'T1': 'T₁',
    'ground motion recorder': 'GMR',
    'tmdi configuration': 'Damper Configuration',
    'tmdi damping ratio': 'Damp. Ratio',
    'tmdi_1 damping': 'cd₁ (N.s/m)',
    'tmdi_1 stiffness': 'kd₁ (N/m)',
    'beta': 'Inertance Ratio',
    'mu': 'Mass Ratio',
    'tmdi frequency ratio': 'Freq. Ratio',
    'tmdi_2 stiffness': 'kd₂ (N/m)',
    'tmdi_2 damping': 'cd₂ (N.s/m)'
}


# === Story-T1 mapping ===
story_t1_map = {
    1: 0.6132,
    2: 0.8392,
    3: 1.1212,
    4: 1.4160,
    5: 1.6335,
    6: 1.8034,
    7: 1.9939,
    8: 2.1107,
    12: 2.8108,
    14: 3.2265,
    15: 3.3478
}

T1_story_map = {v: k for k, v in story_t1_map.items()}

categorical_values = {
    'No.story': list(story_t1_map.keys()),
    'tmdi configuration': ['A', 'B', 'C'],
    'ground motion recorder': list(range(1, 45))
}

numerical_cols = ['T1', 'beta', 'mu', 'tmdi frequency ratio', 'tmdi damping ratio',
                  'tmdi_1 stiffness', 'tmdi_2 stiffness', 'tmdi_1 damping', 'tmdi_2 damping']
categorical_cols = ['No.story', 'tmdi configuration']

models_dir = "models"
targets = [
    'median_pl1_drift', 'beta_total_pl1_drift',
    'median_pl2_drift', 'beta_total_pl2_drift',
    'median_pl3_drift', 'beta_total_pl3_drift',
    'median_pl4_drift', 'beta_total_pl4_drift'
]

# === Fragility function ===
def fragility_curve(median_ds, standard_deviation_ds):
    x_vals = np.linspace(0, 8, 500)
    cdf = stats.norm.cdf((np.log(x_vals) - np.log(median_ds)) / standard_deviation_ds)
    return x_vals, cdf

st.set_page_config(layout="wide")
# === Global style enhancements ===
st.markdown("""
<style>
    /* Page style */
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #fafbfd;
    }

    /* Headers */
    h1, h2, h3, h4 {
        color: #003366;
    }

    /* Tabs */
    .stTabs [role="tablist"] {
        border-bottom: 2px solid #ccc;
    }

    /* Tables */
    thead tr th {
        background-color: #003366;
        color: white;
        font-size: 0.9rem;
    }

    tbody tr td {
        font-size: 0.85rem;
    }

    /* Prediction boxes */
    .highlight-box {
        background-color: #f0f4fa;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 1px 1px 6px rgba(0,0,0,0.05);
        margin-top: 1rem;
    }

    /* Plot padding */
    .element-container:has(.stPlotlyChart) {
        padding: 1rem;
        background-color: #ffffff;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# === Custom App Header ===
st.markdown(
    """
    <div style='text-align: center; padding: 1rem 0; background-color: #f0f2f6; border-radius: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.1);'>
        <h1 style='margin-bottom: 0.2rem; color: #003366;'>ML-Based Response Prediction Tool for SMRFs with TMDI</h1>
        <p style='font-size: 1.1rem; margin-top: 0.1rem; color: #333;'>Developed by Mohammad Reza Hemmati Khollari & Mohammad Sadegh Zare</p>
    </div>
    """,
    unsafe_allow_html=True
)

# st.title("ML-Based Response Prediction Tool for SMRFs with TMDI")
tab1, tab2 = st.tabs(["Seismic Response Prediction", "Fragility Curve Parameters Prediction"])

@st.cache_resource
def load_models():
    models = {}
    for target in targets:
        model_path = os.path.join(models_dir, f"{target}.json")
        if os.path.exists(model_path):
            model = xgb.XGBRegressor()
            model.load_model(model_path)
            models[target] = model

    final_model_path = os.path.join(models_dir, "final_model.json")
    if os.path.exists(final_model_path):
        final_model = xgb.XGBRegressor()
        final_model.load_model(final_model_path)
        models["final_model"] = final_model

    return models

@st.cache_data
def get_label_encoders_full():
    return {
        "tmdi configuration": LabelEncoder().fit(categorical_values['tmdi configuration']),
        "No.story": LabelEncoder().fit(categorical_values['No.story']),
        "ground motion recorder": LabelEncoder().fit(categorical_values['ground motion recorder'])
    }

@st.cache_data
def get_label_encoders():
    return {
        "tmdi configuration": LabelEncoder().fit(categorical_values['tmdi configuration']),
        "No.story": LabelEncoder().fit(categorical_values['No.story'])
    }

models = load_models()
label_encoders_full = get_label_encoders_full()
label_encoders = get_label_encoders()

# === TAB 1: Full Feature Input ===
with tab1:
    st.header("Inputs")

    if "full_inputs" not in st.session_state:
        st.session_state.full_inputs = {}

    col1, col2 = st.columns(2)

    with col1:
        selected_story = st.selectbox("No. Stories", list(story_t1_map.keys()), key="n_story")
        st.session_state.full_inputs['No.story'] = selected_story

    with col2:
        selected_T1 = story_t1_map[selected_story]
        st.session_state.full_inputs['T1'] = selected_T1
        st.markdown(f"### 📦 Estimated T₁: {selected_T1} sec")

    st.session_state.full_inputs['tmdi configuration'] = st.selectbox(
        "Damper Configuration", categorical_values['tmdi configuration'],
        key="tmdi_config_main"
    )
    damper_config = st.session_state.full_inputs['tmdi configuration']

    skip_keys = ['No.story', 'T1', 'tmdi configuration']
    keys = [k for k in feature_name_map.keys() if k not in skip_keys]

    for i in range(0, len(keys), 6):
        cols = st.columns(6)
        for j, col in enumerate(cols):
            if i + j < len(keys):
                key = keys[i + j]
                if damper_config in ['A', 'B'] and key in ['tmdi_2 stiffness', 'tmdi_2 damping']:
                    continue
                label = feature_name_map[key]
                widget_key = f"{key}_full"

                with col:
                    if key in categorical_values:
                        st.session_state.full_inputs[key] = st.selectbox(
                            label, categorical_values[key], key=widget_key)
                    else:
                        min_val, max_val = (0.0, 1.0) if key == 'beta' else (0.0, 0.02) if key == 'mu' else \
                                            (0.0, 1.0) if key == 'tmdi frequency ratio' else (0.0, 0.4) if key == 'tmdi damping ratio' else \
                                            (1e6, 10e6) if key in ['tmdi_1 stiffness', 'tmdi_2 stiffness'] else \
                                            (0.5e6, 5e6) if key in ['tmdi_1 damping', 'tmdi_2 damping'] else (None, None)

                        st.session_state.full_inputs[key] = st.number_input(
                            label,
                            format="%.4f",
                            key=widget_key,
                            min_value=min_val,
                            max_value=max_val
                        )

    if "final_prediction" not in st.session_state:
        st.session_state.final_prediction = None

    if st.button("Predict MIDR"):
        final_input = st.session_state.full_inputs.copy()
        for col, le in label_encoders_full.items():
            if col in final_input:
                final_input[col] = le.transform([final_input[col]])[0]
        final_df = pd.DataFrame([final_input])
        try:
            final_model = models.get("final_model")
            if final_model:
                pred = final_model.predict(final_df)[0]
                st.session_state.final_prediction = round(pred, 6)
            else:
                st.session_state.final_prediction = "Error: final_model not loaded"
        except Exception as e:
            st.session_state.final_prediction = f"Error: {e}"

    if st.session_state.final_prediction is not None:
        st.markdown("### 🧐 Predicted MIDR")
        st.markdown("<div class='highlight-box'>", unsafe_allow_html=True)
        st.table(pd.DataFrame(
            [["Maximum Inter-Story Drift Ratio", st.session_state.final_prediction]],
            columns=["Target", "Value"]
        ))
        st.markdown("</div>", unsafe_allow_html=True)

# === TAB 2: Model Input Only ===
with tab2:
    st.header("Inputs")
    col_left, col_right = st.columns([2, 3])  # Adjust ratio as needed

    with col_left:
        if "model_inputs" not in st.session_state:
            st.session_state.model_inputs = {}

        selected_story_model = st.selectbox("No. Stories", list(story_t1_map.keys()), key="n_story_model")
        st.session_state.model_inputs['No.story'] = selected_story_model

        T1_model = story_t1_map[selected_story_model]
        st.session_state.model_inputs['T1'] = T1_model
        st.markdown(f"### 📦 Estimated T₁: {T1_model} sec")

        damper_config_model = st.selectbox("Damper Configuration", categorical_values['tmdi configuration'], key="damper_model")
        st.session_state.model_inputs['tmdi configuration'] = damper_config_model

        input_keys = [k for k in (categorical_cols + numerical_cols) if k not in ['No.story', 'T1', 'tmdi configuration']]

        for i in range(0, len(input_keys), 3):
            cols = st.columns(3)
            for j, col in enumerate(cols):
                if i + j < len(input_keys):
                    key = input_keys[i + j]
                    if damper_config_model in ['A', 'B'] and key in ['tmdi_2 stiffness', 'tmdi_2 damping']:
                        continue
                    label = feature_name_map[key]
                    widget_key = f"{key}_model"

                    with col:
                        if key in categorical_values:
                            st.session_state.model_inputs[key] = st.selectbox(
                                label, categorical_values[key], key=widget_key
                            )
                        else:
                            min_val, max_val = (0.0, 1.0) if key == 'beta' else (0.0, 0.02) if key == 'mu' else \
                                                (0.0, 1.0) if key == 'tmdi frequency ratio' else (0.0, 0.4) if key == 'tmdi damping ratio' else \
                                                (1e6, 10e6) if key in ['tmdi_1 stiffness', 'tmdi_2 stiffness'] else \
                                                (0.5e6, 5e6) if key in ['tmdi_1 damping', 'tmdi_2 damping'] else (None, None)

                            st.session_state.model_inputs[key] = st.number_input(
                                label,
                                format="%.4f",
                                key=widget_key,
                                min_value=min_val,
                                max_value=max_val
                            )

        if "predictions" not in st.session_state:
            st.session_state.predictions = None

        if st.button("Predict Median and Standard Deviation"):
            encoded_inputs = st.session_state.model_inputs.copy()
            for col, le in label_encoders.items():
                encoded_inputs[col] = le.transform([encoded_inputs[col]])[0]
            input_df = pd.DataFrame([encoded_inputs])

            predictions = {}
            for target, model in models.items():
                if target not in targets:
                    continue
                try:
                    pred = model.predict(input_df)[0]
                    predictions[target] = f"{round(pred, 3):.3f}"
                except Exception as e:
                    predictions[target] = f"Error: {e}"

            st.session_state.predictions = predictions

        if st.session_state.predictions:
            rename_map = {
                "median_pl1_drift": "median_ds1",
                "median_pl2_drift": "median_ds2",
                "median_pl3_drift": "median_ds3",
                "median_pl4_drift": "median_ds4",
                "beta_total_pl1_drift": "standard_deviation_ds1",
                "beta_total_pl2_drift": "standard_deviation_ds2",
                "beta_total_pl3_drift": "standard_deviation_ds3",
                "beta_total_pl4_drift": "standard_deviation_ds4",
            }
            renamed_predictions = {
                rename_map.get(k, k): v for k, v in st.session_state.predictions.items()
            }

            col_plot, col_table = st.columns([2, 1])  # Table and plot side by side

            with col_right:
                if st.session_state.predictions:
                    try:
                        median_ds1 = float(renamed_predictions["median_ds1"])
                        std_dev_ds1 = float(renamed_predictions["standard_deviation_ds1"])
                        median_ds2 = float(renamed_predictions["median_ds2"])
                        std_dev_ds2 = float(renamed_predictions["standard_deviation_ds2"])
                        median_ds3 = float(renamed_predictions["median_ds3"])
                        std_dev_ds3 = float(renamed_predictions["standard_deviation_ds3"])
                        median_ds4 = float(renamed_predictions["median_ds4"])
                        std_dev_ds4 = float(renamed_predictions["standard_deviation_ds4"])

                        x_vals, cdf_ds1 = fragility_curve(median_ds1, std_dev_ds1)
                        _, cdf_ds2 = fragility_curve(median_ds2, std_dev_ds2)
                        _, cdf_ds3 = fragility_curve(median_ds3, std_dev_ds3)
                        _, cdf_ds4 = fragility_curve(median_ds4, std_dev_ds4)

                        col_graph, col_table = st.columns(2)

                        with col_graph:
                            st.markdown("### 📈 Fragility Curves")
                            fig, ax = plt.subplots(figsize=(4, 2.5))
                            ax.plot(x_vals, cdf_ds1, label='DS1 - Slight', linewidth=1.2)
                            ax.plot(x_vals, cdf_ds2, label='DS2 - Moderate', linewidth=1.2)
                            ax.plot(x_vals, cdf_ds3, label='DS3 - Extensive', linewidth=1.2)
                            ax.plot(x_vals, cdf_ds4, label='DS4 - Complete', linewidth=1.2)

                            ax.set_xlim(0, 4.5)
                            ax.set_ylim(0, 1.0)
                            ax.set_xlabel('Sa (g)', fontsize=8)
                            ax.set_ylabel('Probability of Exceedance', fontsize=8)
                            ax.tick_params(axis='both', labelsize=7)
                            ax.legend(fontsize=7)
                            ax.grid(True, which='both', linestyle='--', alpha=0.5)
                            st.pyplot(fig)

                        with col_table:
                            st.markdown("### 🤖 Predicted Median and Logarithmic Standard Deviation")
                            # Add (g) suffix for median_ds keys only in the displayed table
                            display_items = []
                            for key, val in renamed_predictions.items():
                                if key.startswith("median_ds"):
                                    display_items.append((f"{key} (g)", val))
                                else:
                                    display_items.append((key, val))
                            with col_table:
                                st.markdown("<div class='highlight-box'>", unsafe_allow_html=True)
                                display_items = []
                                for key, val in renamed_predictions.items():
                                    if key.startswith("median_ds"):
                                        display_items.append((f"{key} (g)", val))
                                    else:
                                        display_items.append((key, val))
                                st.table(
                                    pd.DataFrame(
                                        display_items,
                                        columns=["Target", "Prediction"]
                                    )
                                )
                                st.markdown("</div>", unsafe_allow_html=True)

                    except Exception as e:
                        st.warning(f"Could not plot fragility curves: {e}")
