import streamlit as st
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import matplotlib as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# --- 1. Physics & Models ---
R_GAS = 8.314

def get_water_saturation_pressure(temp_k):
    """Antoine Equation for Water P0 (bar)"""
    A, B, C = 5.20389, 1733.926, -39.485
    log_p = A - (B / (temp_k + C))
    return 10**log_p

def adjust_affinity(val_ref, T, T_ref, Qst_kj_mol):
    """Clausius-Clapeyron adjustment"""
    Qst_j_mol = Qst_kj_mol * 1000
    exponent = (Qst_j_mol / R_GAS) * (1/T - 1/T_ref)
    return val_ref * np.exp(exponent)

def gab_model(pressure, q_m, C_adj, K_const, P_sat):
    """GAB Model for Water"""
    aw = pressure / P_sat
    aw = np.minimum(aw, 0.9999)
    numerator = q_m * C_adj * K_const * aw
    denominator = (1 - K_const * aw) * (1 - K_const * aw + C_adj * K_const * aw)
    return numerator / denominator

def dual_site_langmuir(pressure, q_m1, b1_adj, q_m2, b2_adj):
    """Dual-Site Langmuir for CO2"""
    site_1 = (q_m1 * b1_adj * pressure) / (1 + b1_adj * pressure)
    site_2 = (q_m2 * b2_adj * pressure) / (1 + b2_adj * pressure)
    return site_1 + site_2

# --- 2. INITIAL DATABASE & SESSION STATE ---

# We define the hardcoded data first
DEFAULT_DATA = {
    "CALF-20": {
        "H2O": {"q_m": 2.0, "C": 2.0, "K": 0.1, "Qst": 15.0},
        "CO2": {"q_m1": 4.5, "b1": 15.0, "q_m2": 2.0, "b2": 0.8, "Qst": 38.0}
    },
    "UiO-66-NH2": {
        "H2O": {"q_m": 14.0, "C": 8.0, "K": 0.85, "Qst": 48.0},
        "CO2": {"q_m1": 6.5, "b1": 4.0, "q_m2": 2.5, "b2": 0.4, "Qst": 35.0}
    },
    "Mg-MOF-74": {
        "H2O": {"q_m": 30.0, "C": 20.0, "K": 0.95, "Qst": 60.0},
        "CO2": {"q_m1": 25.0, "b1": 8.0, "q_m2": 5.0, "b2": 0.5, "Qst": 42.0}
    },
    "HKUST-1": {
        "H2O": {"q_m": 20.0, "C": 10.0, "K": 0.9, "Qst": 50.0},
        "CO2": {"q_m1": 8.0, "b1": 2.5, "q_m2": 4.0, "b2": 0.2, "Qst": 25.0}
    },
    "UiO-66": {
        "H2O": {"q_m": 10.0, "C": 5.0, "K": 0.8, "Qst": 45.0},
        "CO2": {"q_m1": 4.0, "b1": 1.5, "q_m2": 3.0, "b2": 0.3, "Qst": 24.0}
    },
    "ZIF-8": {
        "H2O": {"q_m": 1.0, "C": 0.1, "K": 0.2, "Qst": 10.0},
        "CO2": {"q_m1": 12.0, "b1": 0.5, "q_m2": 0.0, "b2": 0.0, "Qst": 18.0}
    },
}

# Initialize Session State
if 'mof_data' not in st.session_state:
    st.session_state['mof_data'] = DEFAULT_DATA.copy()

# --- 3. Streamlit Interface ---

st.set_page_config(page_title="MOF Selector V4", layout="wide")
st.title("📊 MOF Adsorbent Selector & Designer")

# Add the new tab at the beginning
tab_add, tab_rec, tab_co2, tab_h2o = st.tabs(["➕ Add Custom Material", "🎯 Recommendation Engine", "☁️ CO2 Performance", "💧 Water Stability"])

# --- TAB 1: ADD CUSTOM MATERIAL ---
with tab_add:
    st.header("Design Your Own Adsorbent")
    st.markdown("Input model parameters to simulate a new material (e.g., your **Y2030-1**).")
   
    with st.form("new_material_form"):
        col_meta, col_h2o, col_co2 = st.columns(3)
       
        with col_meta:
            st.subheader("1. Identity")
            new_name = st.text_input("Material Name", value="Y2030-1")
            st.info("Parameters below are pre-filled with **Double UiO-66-NH2** values as requested.")
           
        with col_h2o:
            st.subheader("2. Water (GAB Model)")
            # Pre-filled with Y2030-1 values (Double UiO-66-NH2)
            n_qm_w = st.number_input("Max Capacity q_m (mmol/g)", value=28.0)
            n_C = st.number_input("GAB C Constant", value=8.0)
            n_K = st.number_input("GAB K Constant", value=0.85)
            n_Qst_w = st.number_input("Water Heat of Ads (kJ/mol)", value=48.0)
           
        with col_co2:
            st.subheader("3. CO2 (Dual-Site Langmuir)")
            # Pre-filled with Y2030-1 values (Double UiO-66-NH2)
            n_qm1 = st.number_input("Site 1 Capacity q_m1", value=13.0)
            n_b1 = st.number_input("Site 1 Affinity b1", value=4.0)
            n_qm2 = st.number_input("Site 2 Capacity q_m2", value=5.0)
            n_b2 = st.number_input("Site 2 Affinity b2", value=0.4)
            n_Qst_c = st.number_input("CO2 Heat of Ads (kJ/mol)", value=35.0)
           
        submit = st.form_submit_button("Add Material to Database")
       
        if submit:
            # Construct the dictionary entry
            new_entry = {
                "H2O": {"q_m": n_qm_w, "C": n_C, "K": n_K, "Qst": n_Qst_w},
                "CO2": {"q_m1": n_qm1, "b1": n_b1, "q_m2": n_qm2, "b2": n_b2, "Qst": n_Qst_c}
            }
            # Update Session State
            st.session_state['mof_data'][new_name] = new_entry
            st.success(f"Successfully added **{new_name}**! Go to the 'Recommendation' or 'Performance' tabs to see how it ranks.")

# --- TAB 2: RECOMMENDATION ENGINE ---
with tab_rec:
    st.header("Find the Best Adsorbent")
   
    # Input Section
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1:
        max_regen_temp = st.slider("Max Regeneration Temp Available (°C)", 60, 200, 100)
    with col_in2:
        user_temp = st.number_input("Adsorption Temp (K)", 250, 400, 298)
    with col_in3:
        target_co2_mbar = st.number_input("Target CO2 Pressure (mbar)", 0.1, 5000.0, 100.0)

    st.markdown("---")

    # Calculation using SESSION STATE data
    results = []
    target_p_bar = target_co2_mbar / 1000.0
   
    # Loop through the DYNAMIC database
    for mof_name, data in st.session_state['mof_data'].items():
        c_p = data["CO2"]
       
        # Calculate Capacity at Adsorption T
        b1_adj = adjust_affinity(c_p["b1"], user_temp, 298, c_p["Qst"])
        b2_adj = adjust_affinity(c_p["b2"], user_temp, 298, c_p["Qst"])
        cap = dual_site_langmuir(target_p_bar, c_p["q_m1"], b1_adj, c_p["q_m2"], b2_adj)
       
        suitability = "✅ Excellent"
        if c_p["Qst"] > 40:
            suitability = "⚠️ Hard Regen (>100C likely)"
        elif mof_name in ["HKUST-1", "Mg-MOF-74"]:
            suitability = "⚠️ Water Unstable"
        if max_regen_temp < 100 and c_p["Qst"] > 38:
             suitability = "❌ Requires High T"

        results.append({
            "MOF": mof_name,
            "Capacity": cap,
            "Qst (kJ/mol)": c_p["Qst"],
            "Regen/Stability Status": suitability
        })
   
    df_res = pd.DataFrame(results).sort_values(by="Capacity", ascending=False)

    col_res1, col_res2 = st.columns([2, 1])
    with col_res1:
        st.subheader("Ranking Results")
        # Check if Y2030-1 is top
        if not df_res.empty and df_res.iloc[0]['MOF'] == "Y2030-1":
            st.success("🎉 Your custom material **Y2030-1** is the top performer!")
           
        st.dataframe(
            df_res.style.background_gradient(subset=["Capacity"], cmap="Greens"),
            hide_index=True,
            column_config={"Capacity": st.column_config.NumberColumn("Capacity (mmol/g)", format="%.2f")}
        )
    with col_res2:
        st.info("Tip: If you just added a custom material, it will appear in this list immediately.")

# --- TAB 3: CO2 PLOTS ---
with tab_co2:
    st.subheader("CO2 Isotherms (Log Scale)")
    fig_co2, ax_co2 = plt.subplots(figsize=(10, 5))
    p_range = np.logspace(np.log10(0.0001), np.log10(1.0), 200)

    # Use Dynamic Colors
    colors = sns.color_palette("bright", len(st.session_state['mof_data']))
   
    for i, (mof_name, data) in enumerate(st.session_state['mof_data'].items()):
        params = data["CO2"]
        b1_adj = adjust_affinity(params["b1"], user_temp, 298, params["Qst"])
        b2_adj = adjust_affinity(params["b2"], user_temp, 298, params["Qst"])
       
        q_curve = dual_site_langmuir(p_range, params["q_m1"], b1_adj, params["q_m2"], b2_adj)
       
        # Highlight logic
        style = '--'
        width = 1.5
        if mof_name in ["CALF-20", "UiO-66-NH2"]:
            style = '-'
            width = 2.5
        if mof_name == "Y2030-1":
            style = '-.'
            width = 3.0 # Make user custom material very visible
       
        ax_co2.plot(p_range * 1000, q_curve, label=mof_name, linestyle=style, linewidth=width, color=colors[i])
       
    ax_co2.set_xscale('log')
    ax_co2.set_xlabel("Pressure (mbar)")
    ax_co2.set_ylabel("Uptake (mmol/g)")
    ax_co2.legend()
    ax_co2.grid(True, which="both", linestyle='--', alpha=0.5)
    st.pyplot(fig_co2)

# --- TAB 4: WATER STABILITY CHECK ---
with tab_h2o:
    st.subheader("Water Isotherms")
    fig_h2o, ax_h2o = plt.subplots(figsize=(10, 5))
   
    P_sat_std = get_water_saturation_pressure(298)
    rh_range = np.linspace(0.01, 0.9, 200)
    p_range_bar = rh_range * P_sat_std

    colors = sns.color_palette("bright", len(st.session_state['mof_data']))

    for i, (mof_name, data) in enumerate(st.session_state['mof_data'].items()):
        params = data["H2O"]
        q_curve = gab_model(p_range_bar, params["q_m"], params["C"], params["K"], P_sat_std)
        ax_h2o.plot(rh_range*100, q_curve, label=mof_name, color=colors[i])
       
    ax_h2o.set_xlabel("Relative Humidity (%)")
    ax_h2o.set_ylabel("Water Uptake (mmol/g)")
    ax_h2o.legend()

    st.pyplot(fig_h2o)
