import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import datetime
import os  # NEW: To access your local hard drive

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

# --- 2. SEARCH FUNCTIONS (UPDATED) ---

def search_crossref(keyword="MOF adsorption", rows=5):
    """Searches the Web (Crossref API)"""
    current_year = datetime.datetime.now().year
    url = "https://na01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fapi.crossref.org%2Fworks&data=05%7C02%7C%7C2597a336fa9f46a382af08de4979de2f%7C84df9e7fe9f640afb435aaaaaaaaaaaa%7C1%7C0%7C639028984918825053%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&sdata=8jIL5drguGEIoJIZTnnXmTVor9Jayk7vi%2B2CFB39CEE%3D&reserved=0"
    params = {
        "query": keyword,
        "filter": f"from-pub-date:{current_year-1}-01-01",
        "rows": rows,
        "sort": "published",
        "order": "desc"
    }
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            items = response.json()["message"]["items"]
            results = []
            for item in items:
                title = item.get("title", ["No Title"])[0]
                link = item.get("URL", f"https://na01.safelinks.protection.outlook.com/?url=https%3A%2F%2Fdoi.org%2F&data=05%7C02%7C%7C2597a336fa9f46a382af08de4979de2f%7C84df9e7fe9f640afb435aaaaaaaaaaaa%7C1%7C0%7C639028984918851528%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&sdata=VYAjQ9TXTCPMC%2BbkhAYtxxXtBdYjMdoRUCSofJhezdA%3D&reserved=0{item.get('DOI', '')}")
                journal = item.get("container-title", ["Unknown Journal"])[0]
                results.append({"Title": title, "Journal": journal, "Link": link, "Source": "🌐 Web"})
            return results
        return []
    except:
        return []

def search_local_files(directory, keyword):
    """Searches Local Folder for filenames containing the keyword"""
    results = []
    # Check if directory exists
    if not os.path.isdir(directory):
        return [{"Title": "Error: Folder not found", "Journal": "Check Path", "Link": "#", "Source": "❌"}]
   
    # Walk through the folder
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            # We look for PDFs that match the keyword
            if file.lower().endswith(".pdf") and keyword.lower() in file.lower():
                full_path = os.path.join(root, file)
                results.append({
                    "Title": file,  # Use filename as title
                    "Journal": "Local File",
                    "Link": full_path,
                    "Source": "📂 Local"
                })
                count += 1
                if count >= 10: break # Limit local results to 10 to prevent freezing
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

# --- 4. APP LAYOUT ---
st.set_page_config(page_title="MOF Discovery V6", layout="wide")
st.title("🌐 MOF Discovery & Analysis Platform")

tab_feed, tab_add, tab_rec, tab_co2, tab_h2o = st.tabs([
    "📚 Literature Feed", "➕ Add New Data", "🎯 Recommendation Engine", "☁️ CO2 Performance", "💧 Water Stability"
])

# --- TAB 1: LITERATURE FEED (HYBRID) ---
with tab_feed:
    st.header("Hybrid Literature Scanner")
    st.markdown("Search both the **Web** (Crossref) and your **Local Computer** for relevant papers.")
   
    col_search, col_results = st.columns([1, 2])
   
    with col_search:
        st.subheader("Search Settings")
        query_text = st.text_input("Keyword", value="MOF")
       
        # New: Local Folder Input
        search_mode = st.radio("Search Source:", ["Web Only", "Local Only", "Hybrid (Both)"])
       
        local_path = ""
        if "Local" in search_mode or "Hybrid" in search_mode:
            st.markdown("---")
            local_path = st.text_input("Local Folder Path",
                                       value=r"C:\Users\Public\Documents",
                                       help="Copy paste your folder path here. E.g., C:\Papers")
       
        if st.button("🔎 Scan Library"):
            found_papers = []
           
            # 1. Search Local
            if "Local" in search_mode or "Hybrid" in search_mode:
                if local_path:
                    with st.spinner("Scanning Local Drive..."):
                        found_papers.extend(search_local_files(local_path, query_text))
           
            # 2. Search Web
            if "Web" in search_mode or "Hybrid" in search_mode:
                with st.spinner("Scanning Web..."):
                    found_papers.extend(search_crossref(query_text, rows=5))
           
            st.session_state['found_papers'] = found_papers

    with col_results:
        st.subheader("Search Results")
        if 'found_papers' in st.session_state and st.session_state['found_papers']:
            for idx, paper in enumerate(st.session_state['found_papers']):
                # Visual distinction between Local and Web
                icon = paper.get("Source", "❓")
               
                with st.expander(f"{icon} {paper['Title']}"):
                    st.caption(f"Source: {paper['Journal']} | Path/Link: {paper['Link']}")
                   
                    # Draft Button
                    if st.button(f"Draft to Database", key=f"btn_{idx}"):
                        # Clean up title for database name
                        clean_name = paper['Title'][:20].replace(" ", "_")
                        st.session_state['draft_name'] = clean_name
                        st.success(f"Copied '{clean_name}' to Add Tab!")
        else:
            st.info("No papers found. Check your keyword or folder path.")

# --- TAB 2: ADD DATA ---
with tab_add:
    st.header("Expand the Database")
    draft_val = st.session_state.get('draft_name', "New_MOF_Year")
   
    with st.form("new_material_form"):
        col_meta, col_h2o, col_co2 = st.columns(3)
        with col_meta:
            st.subheader("1. Identity")
            new_name = st.text_input("Material Name", value=draft_val)
        with col_h2o:
            st.subheader("2. Water Params")
            n_qm_w = st.number_input("Max Capacity q_m", value=10.0)
            n_C = st.number_input("GAB C", value=5.0)
            n_K = st.number_input("GAB K", value=0.8)
            n_Qst_w = st.number_input("Heat of Ads (kJ/mol)", value=40.0)
        with col_co2:
            st.subheader("3. CO2 Params")
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

# --- TAB 4 & 5: PLOTS ---
with tab_co2:
    st.subheader("CO2 Isotherms")
    fig, ax = plt.subplots(figsize=(10, 5))
    p_range = np.logspace(np.log10(0.0001), np.log10(1.0), 200)
    colors = sns.color_palette("bright", len(st.session_state['mof_data']))
    for i, (name, data) in enumerate(st.session_state['mof_data'].items()):
        p = data["CO2"]
        b1 = adjust_affinity(p["b1"], user_temp, 298, p["Qst"])
        b2 = adjust_affinity(p["b2"], user_temp, 298, p["Qst"])
        ax.plot(p_range*1000, dual_site_langmuir(p_range, p["q_m1"], b1, p["q_m2"], b2), label=name, color=colors[i])
    ax.set_xscale('log'); ax.legend(); st.pyplot(fig)

with tab_h2o:
    st.subheader("Water Isotherms")
    fig, ax = plt.subplots(figsize=(10, 5))
    rh = np.linspace(0.01, 0.9, 200)
    colors = sns.color_palette("bright", len(st.session_state['mof_data']))
    P_sat = get_water_saturation_pressure(298)
    for i, (name, data) in enumerate(st.session_state['mof_data'].items()):
        p = data["H2O"]
        ax.plot(rh*100, gab_model(rh*P_sat, p["q_m"], p["C"], p["K"], P_sat), label=name, color=colors[i])
    ax.legend(); st.pyplot(fig)
