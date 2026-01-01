import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go # NEW: Interactive Plotting Library
import requests
import datetime
import os
import io

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

# --- 2. SEARCH FUNCTIONS ---
def search_crossref(keyword="MOF adsorption", rows=5):
    """Searches the Web (Crossref API)"""
    current_year = datetime.datetime.now().year
    url = "https://na01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fapi.crossref.org%2Fworks&data=05%7C02%7C%7C87a9e9c2999a41c2e24308de497ec9ab%7C84df9e7fe9f640afb435aaaaaaaaaaaa%7C1%7C0%7C639029006052963317%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&sdata=ZsFTw7y5SZ96xSFgn9LnvaZ4BUz4kGRQEBTzw4bmrQU%3D&reserved=0"
    params = {
        "query": keyword, "filter": f"from-pub-date:{current_year-1}-01-01",
        "rows": rows, "sort": "published", "order": "desc"
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            items = response.json()["message"]["items"]
            results = []
            for item in items:
                title = item.get("title", ["No Title"])[0]
                link = item.get("URL", f"https://na01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fdoi.org%2F&data=05%7C02%7C%7C87a9e9c2999a41c2e24308de497ec9ab%7C84df9e7fe9f640afb435aaaaaaaaaaaa%7C1%7C0%7C639029006052979944%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&sdata=vC5QCRsuZWVPYTTjOYC9Bvy4erovXqP0wVK%2Bvs4gPiM%3D&reserved=0{item.get('DOI', '')}")
                journal = item.get("container-title", ["Unknown Journal"])[0]
                results.append({"Title": title, "Journal": journal, "Link": link, "Source": "🌐 Web"})
            return results
        return []
    except: return []

def search_local_files(directory, keyword):
    """Searches Local Folder"""
    results = []
    if not os.path.isdir(directory): return []
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pdf") and keyword.lower() in file.lower():
                results.append({"Title": file, "Journal": "Local File", "Link": os.path.join(root, file), "Source": "📂 Local"})
                count += 1
                if count >= 10: break
    return results

# --- 3. DATABASE ---
DEFAULT_DATA = {
    "CALF-20": {"H2O": {"q_m": 2.0, "C": 2.0, "K": 0.1, "Qst": 15.0}, "CO2": {"q_m1": 4.5, "b1": 15.0, "q_m2": 2.0, "b2": 0.8, "Qst": 38.0}},
    "UiO-66-NH2": {"H2O": {"q_m": 14.0, "C": 8.0, "K": 0.85, "Qst": 48.0}, "CO2": {"q_m1": 6.5, "b1": 4.0, "q_m2": 2.5, "b2": 0.4, "Qst": 35.0}},
    "Mg-MOF-74": {"H2O": {"q_m": 30.0, "C": 20.0, "K": 0.95, "Qst": 60.0}, "CO2": {"q_m1": 25.0, "b1": 8.0, "q_m2": 5.0, "b2": 0.5, "Qst": 42.0}},
    "HKUST-1": {"H2O": {"q_m": 20.0, "C": 10.0, "K": 0.9, "Qst": 50.0}, "CO2": {"q_m1": 8.0, "b1": 2.5, "q_m2": 4.0, "b2": 0.2, "Qst": 25.0}},
    "UiO-66": {"H2O": {"q_m": 10.0, "C": 5.0, "K": 0.8, "Qst": 45.0}, "CO2": {"q_m1": 4.0, "b1": 1.5, "q_m2": 3.0, "b2": 0.3, "Qst": 24.0}},
    "ZIF-8": {"H2O": {"q_m": 1.0, "C": 0.1, "K": 0.2, "Qst": 10.0}, "CO2": {"q_m1": 12.0, "b1": 0.5, "q_m2": 0.0, "b2": 0.0, "Qst": 18.0}},
}

if 'mof_data' not in st.session_state:
    st.session_state['mof_data'] = DEFAULT_DATA.copy()

# --- 4. EXPORT FUNCTION ---
def convert_db_to_excel():
    data_list = []
    for name, params in st.session_state['mof_data'].items():
        row = {"Material Name": name}
        for k, v in params["H2O"].items(): row[f"H2O_{k}"] = v
        for k, v in params["CO2"].items(): row[f"CO2_{k}"] = v
        data_list.append(row)
    df = pd.DataFrame(data_list)
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Sorbents')
    return output.getvalue()

# --- 5. APP LAYOUT ---
st.set_page_config(page_title="MOF Discovery V8", layout="wide")

# SIDEBAR EXPORT
st.sidebar.title("💾 Data Management")
excel_data = convert_db_to_excel()
st.sidebar.download_button("Download Database (Excel)", excel_data, 'my_sorbent_database.xlsx')

st.title("🌐 MOF Discovery & Analysis Platform")

tab_feed, tab_add, tab_rec, tab_co2, tab_h2o = st.tabs([
    "📚 Literature Feed", "➕ Add New Data", "🎯 Recommendation Engine", "☁️ CO2 Performance", "💧 Water Stability"
])

# --- TAB 1: LITERATURE FEED ---
with tab_feed:
    st.header("Hybrid Literature Scanner")
    col_search, col_results = st.columns([1, 2])
    with col_search:
        query_text = st.text_input("Keyword", value="MOF")
        search_mode = st.radio("Search Source:", ["Web Only", "Local Only", "Hybrid (Both)"])
        local_path = ""
        if "Local" in search_mode or "Hybrid" in search_mode:
            local_path = st.text_input("Local Folder Path", value=r"C:\Users\Public\Documents")
       
        if st.button("🔎 Scan Library"):
            found_papers = []
            if "Local" in search_mode or "Hybrid" in search_mode:
                if local_path: found_papers.extend(search_local_files(local_path, query_text))
            if "Web" in search_mode or "Hybrid" in search_mode:
                found_papers.extend(search_crossref(query_text, rows=5))
            st.session_state['found_papers'] = found_papers

    with col_results:
        if 'found_papers' in st.session_state and st.session_state['found_papers']:
            for idx, paper in enumerate(st.session_state['found_papers']):
                icon = paper.get("Source", "❓")
                with st.expander(f"{icon} {paper['Title']}"):
                    st.caption(f"Source: {paper['Journal']} | Path/Link: {paper['Link']}")
                    if st.button(f"Draft to Database", key=f"btn_{idx}"):
                        clean_name = paper['Title'][:20].replace(" ", "_")
                        st.session_state['draft_name'] = clean_name
                        st.success(f"Copied '{clean_name}' to Add Tab!")
        else: st.info("No papers found.")

# --- TAB 2: ADD DATA ---
with tab_add:
    st.header("Expand the Database")
    draft_val = st.session_state.get('draft_name', "New_MOF_Year")
    with st.form("new_material_form"):
        col_meta, col_h2o, col_co2 = st.columns(3)
        with col_meta: new_name = st.text_input("Material Name", value=draft_val)
        with col_h2o:
            st.subheader("Water Params")
            n_qm_w = st.number_input("Max Capacity q_m", value=10.0)
            n_C = st.number_input("GAB C", value=5.0)
            n_K = st.number_input("GAB K", value=0.8)
            n_Qst_w = st.number_input("Heat of Ads (kJ/mol)", value=40.0)
        with col_co2:
            st.subheader("CO2 Params")
            n_qm1 = st.number_input("Site 1 q_m", value=5.0)
            n_b1 = st.number_input("Site 1 b", value=2.0)
            n_qm2 = st.number_input("Site 2 q_m", value=1.0)
            n_b2 = st.number_input("Site 2 b", value=0.1)
            n_Qst_c = st.number_input("Heat of Ads (kJ/mol)", value=30.0)
           
        if st.form_submit_button("Add to Database"):
            st.session_state['mof_data'][new_name] = {
                "H2O": {"q_m": n_qm_w, "C": n_C, "K": n_K, "Qst": n_Qst_w},
                "CO2": {"q_m1": n_qm1, "b1": n_b1, "q_m2": n_qm2, "b2": n_b2, "Qst": n_Qst_c}
            }
            st.success(f"Added **{new_name}**!")

# --- TAB 3: RECOMMENDATION ---
with tab_rec:
    st.header("Recommendation Engine")
    col_in1, col_in2, col_in3 = st.columns(3)
    with col_in1: max_regen_temp = st.slider("Max Regen Temp (°C)", 60, 200, 100)
    with col_in2: user_temp = st.number_input("Operating Temp (K)", 250, 400, 298)
    with col_in3: target_co2_mbar = st.number_input("CO2 Partial Pressure (mbar)", 0.1, 5000.0, 100.0)
    st.markdown("---")
   
    results = []
    target_p_bar = target_co2_mbar / 1000.0
    for mof_name, data in st.session_state['mof_data'].items():
        c_p = data["CO2"]
        b1_adj = adjust_affinity(c_p["b1"], user_temp, 298, c_p["Qst"])
        b2_adj = adjust_affinity(c_p["b2"], user_temp, 298, c_p["Qst"])
        cap = dual_site_langmuir(target_p_bar, c_p["q_m1"], b1_adj, c_p["q_m2"], b2_adj)
        suitability = "✅ Excellent"
        if c_p["Qst"] > 40: suitability = "⚠️ Hard Regen"
        if max_regen_temp < 100 and c_p["Qst"] > 38: suitability = "❌ Requires High T"
        results.append({"MOF": mof_name, "Capacity": cap, "Qst": c_p["Qst"], "Status": suitability})
   
    st.dataframe(pd.DataFrame(results).sort_values(by="Capacity", ascending=False).style.background_gradient(subset=["Capacity"], cmap="Greens"), hide_index=True)

# --- TAB 4: CO2 INTERACTIVE PLOTS (UPDATED) ---
with tab_co2:
    st.subheader("CO2 Isotherms (Interactive)")
    st.caption("Hover over lines to see values. Click legend to hide/show materials.")
   
    fig_co2 = go.Figure()
   
    # Generate data
    p_range = np.logspace(np.log10(0.0001), np.log10(1.0), 200) # 0.1 mbar to 1 bar
   
    for mof_name, data in st.session_state['mof_data'].items():
        params = data["CO2"]
        b1 = adjust_affinity(params["b1"], user_temp, 298, params["Qst"])
        b2 = adjust_affinity(params["b2"], user_temp, 298, params["Qst"])
       
        y_vals = dual_site_langmuir(p_range, params["q_m1"], b1, params["q_m2"], b2)
       
        # Add Trace
        fig_co2.add_trace(go.Scatter(
            x=p_range * 1000, # Convert bar to mbar
            y=y_vals,
            mode='lines',
            name=mof_name,
            hovertemplate='%{y:.2f} mmol/g @ %{x:.1f} mbar'
        ))
       
    # Styling
    fig_co2.update_layout(
        xaxis_type="log",
        xaxis_title="Pressure (mbar)",
        yaxis_title="Uptake (mmol/g)",
        hovermode="x unified",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig_co2, use_container_width=True)

# --- TAB 5: WATER INTERACTIVE PLOTS (UPDATED) ---
with tab_h2o:
    st.subheader("Water Isotherms (Interactive)")
    st.caption("Check for S-shape behavior indicating pore condensation.")
   
    fig_h2o = go.Figure()
   
    rh_range = np.linspace(0.01, 0.9, 200)
    P_sat = get_water_saturation_pressure(298)
   
    for mof_name, data in st.session_state['mof_data'].items():
        params = data["H2O"]
        # Use GAB
        y_vals = gab_model(rh_range*P_sat, params["q_m"], params["C"], params["K"], P_sat)
       
        fig_h2o.add_trace(go.Scatter(
            x=rh_range*100, # Convert to %
            y=y_vals,
            mode='lines',
            name=mof_name,
            hovertemplate='%{y:.2f} mmol/g @ %{x:.0f}% RH'
        ))
       
    fig_h2o.update_layout(
        xaxis_title="Relative Humidity (%)",
        yaxis_title="Uptake (mmol/g)",
        hovermode="x unified",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig_h2o, use_container_width=True)

