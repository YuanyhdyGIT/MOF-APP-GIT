import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import json
from scipy.optimize import curve_fit

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="MOF Omni-Tool V17", layout="wide")

# --- 2. SAFE IMPORTS ---
try:
    import google.generativeai as genai
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False

try:
    from stmol import showmol
    import py3Dmol
    VIZ_AVAILABLE = True
except ImportError:
    VIZ_AVAILABLE = False

try:
    import pypdf
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# --- 3. PHYSICS & MODELS ---
R_GAS = 8.314

def get_water_saturation_pressure(temp_k):
    """Antoine Equation for Water Psat (bar)"""
    A, B, C = 5.20389, 1733.926, -39.485
    if temp_k + C == 0: return 1.0 
    log_p = A - (B / (temp_k + C))
    return (10**log_p) 

def adjust_affinity(val_ref, T, T_ref, Qst_kj_mol):
    """Clausius-Clapeyron adjustment"""
    Qst_j_mol = Qst_kj_mol * 1000
    if T == 0: return val_ref
    exponent = (Qst_j_mol / R_GAS) * (1/T - 1/T_ref)
    return val_ref * np.exp(exponent)

def gab_model(pressure, q_m, C_adj, K_const, P_sat):
    """GAB Model for Water"""
    if P_sat == 0: return 0
    aw = pressure / P_sat 
    aw = np.minimum(aw, 0.99) # Limit to 99% RH
    numerator = q_m * C_adj * K_const * aw
    denominator = (1 - K_const * aw) * (1 - K_const * aw + C_adj * K_const * aw)
    return numerator / denominator

def dual_site_langmuir(pressure, q_m1, b1_adj, q_m2, b2_adj):
    """Dual-Site Langmuir"""
    site_1 = (q_m1 * b1_adj * pressure) / (1 + b1_adj * pressure)
    site_2 = (q_m2 * b2_adj * pressure) / (1 + b2_adj * pressure)
    return site_1 + site_2

def simulate_kinetic_step(q_initial, q_target, k, time_points):
    """LDF kinetics"""
    return q_target + (q_initial - q_target) * np.exp(-k * time_points)

# --- 4. DATA PROCESSING ---
def extract_text_from_pdf(uploaded_file):
    if not PDF_AVAILABLE: return "Error: pypdf library not installed."
    try:
        reader = pypdf.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return str(e)

def extract_data_with_gemini(text_chunk, api_key):
    if not AI_AVAILABLE: return {"Error": "Library google-generativeai not installed."}
    
    genai.configure(api_key=api_key)
    # UPDATED: Use gemini-pro for stability
    try:
        model = genai.GenerativeModel('gemini-pro')
    except:
        return {"Error": "Model not available."}

    prompt = f"""
    Extract MOF adsorption parameters. Return ONLY raw JSON (no markdown).
    
    Structure: {{
        "Material_Name": "String",
        "H2O": {{"q_m": float, "C": float, "K": float, "Qst": float}},
        "CO2": {{"q_m1": float, "b1": float, "q_m2": float, "b2": float, "Qst": float}},
        "CIF_URL": "String or null"
    }}
    Text: {text_chunk[:25000]} 
    """ 
    try:
        response = model.generate_content(prompt)
        clean = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception as e:
        return {"Error": str(e)}

# --- 5. DATA ---
DEFAULT_DATA = {
    "CALF-20": {
        "H2O": {"q_m": 2.0, "C": 2.0, "K": 0.1, "Qst": 15.0}, 
        "CO2": {"q_m1": 4.5, "b1": 15.0, "q_m2": 2.0, "b2": 0.8, "Qst": 38.0},
        "cif_url": "https://raw.githubusercontent.com/hirstgroup/hirstgroup.github.io/master/cifs/MOF-5.cif" 
    },
    "HKUST-1": {
        "H2O": {"q_m": 25.0, "C": 5.0, "K": 0.9, "Qst": 20.0},
        "CO2": {"q_m1": 10.0, "b1": 0.8, "q_m2": 0.0, "b2": 0.0, "Qst": 25.0}, 
        "cif_url": "https://raw.githubusercontent.com/hirstgroup/hirstgroup.github.io/master/cifs/HKUST-1.cif"
    },
    "UiO-66-NH2": {
        "H2O": {"q_m": 12.0, "C": 10.0, "K": 0.8, "Qst": 45.0},
        "CO2": {"q_m1": 6.0, "b1": 0.9, "q_m2": 0.0, "b2": 0.0, "Qst": 28.0},
        "cif_url": "https://raw.githubusercontent.com/hirstgroup/hirstgroup.github.io/master/cifs/UiO-66.cif"
    },
    "Mg-MOF-74": {
         "H2O": {"q_m": 35.0, "C": 15.0, "K": 0.95, "Qst": 50.0},
         "CO2": {"q_m1": 30.0, "b1": 2.5, "q_m2": 0.0, "b2": 0.0, "Qst": 42.0},
         "cif_url": "https://raw.githubusercontent.com/hirstgroup/hirstgroup.github.io/master/cifs/Mg-MOF-74.cif"
    }
}

if 'mof_data' not in st.session_state:
    st.session_state['mof_data'] = DEFAULT_DATA.copy()

if 'editor_draft' not in st.session_state:
    st.session_state['editor_draft'] = {}

# --- 6. APP UI ---
st.sidebar.title("⚙️ Controls")
api_key = st.sidebar.text_input("Google API Key (Optional)", type="password")

st.title("⚗️ MOF Omni-Tool V17")

tabs = st.tabs([
    "🤖 AI Extraction", 
    "📚 PDF Scanner", 
    "➕ Add Data", 
    "🎯 Rec. Engine", 
    "📊 Isotherms", 
    "🧊 Structure",
    "🔄 Kinetics" 
])

# --- TAB 1: AI ---
with tabs[0]:
    st.header("1. Intelligence Hub")
    pdf_txt = st.session_state.get('pdf_text', '')
    col_in, col_res = st.columns(2)
    with col_in:
        data_source = st.radio("Input Source", ["Manual Text Paste", "Scanned PDF (from Tab 2)"], 
                               index=1 if pdf_txt else 0)
        
        if data_source == "Scanned PDF (from Tab 2)":
            if pdf_txt:
                st.success(f"Files loaded! ({len(pdf_txt)} chars)")
                final_text = pdf_txt
            else:
                st.warning("No PDF scanned yet. Please upload in Tab 2 first.")
                final_text = ""
        else:
            final_text = st.text_area("Paste Abstract/Text:", height=150)
            
        if st.button("🚀 Analyze Text"):
            if not api_key: st.error("No API Key")
            elif not final_text: st.error("No text to analyze")
            else:
                with st.spinner("Gemini is reading..."):
                    res = extract_data_with_gemini(final_text, api_key)
                    if "Error" not in res: 
                        st.session_state['ai_result'] = res
                        st.success("Extraction Complete!")
                    else:
                        st.error(res["Error"])

    with col_res:
        if 'ai_result' in st.session_state:
            st.subheader("AI Findings")
            st.json(st.session_state['ai_result'])
            if st.button("➡️ Transfer to Database Editor (Tab 3)"):
                st.session_state['editor_draft'] = st.session_state['ai_result']
                st.info("Data sent to Tab 3! Go there to save.")

# --- TAB 2: PDF ---
with tabs[1]:
    st.header("2. Document Scanner")
    col_pdf, col_web = st.columns(2)
    with col_pdf:
        st.subheader("Local PDF Upload")
        uploaded_pdf = st.file_uploader("Upload Paper", type=["pdf"])
        if uploaded_pdf:
            if PDF_AVAILABLE:
                with st.spinner("Scanning..."):
                    pdf_text = extract_text_from_pdf(uploaded_pdf)
                    st.session_state['pdf_text'] = pdf_text
                    st.success(f"Scanned {len(pdf_text)} characters.")
            else:
                st.error("Install `pypdf` to read PDF files.")
    with col_web:
        st.subheader("Web Search (CrossRef)")
        kw = st.text_input("Keyword", "MOF adsorption")
        if st.button("Search"):
            url = "https://api.crossref.org/works"
            r = requests.get(url, params={"query": kw, "rows": 5})
            if r.status_code == 200:
                for i in r.json()["message"]["items"]:
                    st.markdown(f"**{i.get('title',['?'])[0]}** [Link]({i.get('URL','')})")

# --- TAB 3: ADD DATA (UPDATED) ---
with tabs[2]:
    st.header("3. Database Editor")
    
    col_tools, col_form = st.columns([1, 2])
    
    with col_tools:
        st.subheader("Auto-Fill Tools")
        if st.button("📂 Load from AI Result"):
            if st.session_state['editor_draft']:
                d = st.session_state['editor_draft']
                st.session_state['f_name'] = d.get("Material_Name", "New")
                c = d.get("CO2", {})
                st.session_state['f_cq1'] = c.get("q_m1", 0.0)
                st.session_state['f_cb1'] = c.get("b1", 0.0)
                st.session_state['f_cqst'] = c.get("Qst", 0.0)
                w = d.get("H2O", {})
                st.session_state['f_wq'] = w.get("q_m", 0.0)
                st.session_state['f_wc'] = w.get("C", 0.0)
                st.success("AI Data Loaded!")
            else:
                st.warning("No AI data found.")
        
        st.markdown("---")
        if st.button("🧪 Load 'Y-2030-1' Template"):
            ref = DEFAULT_DATA["CALF-20"]
            st.session_state['f_name'] = "Y-2030-1"
            st.session_state['f_cq1'] = ref["CO2"]["q_m1"] * 3
            st.session_state['f_cq2'] = ref["CO2"]["q_m2"] * 3
            st.session_state['f_cb1'] = ref["CO2"]["b1"]
            st.session_state['f_cb2'] = ref["CO2"]["b2"]
            st.session_state['f_cqst'] = ref["CO2"]["Qst"]
            st.session_state['f_wq'] = ref["H2O"]["q_m"] * 3
            st.session_state['f_wc'] = ref["H2O"]["C"]
            st.session_state['f_wk'] = ref["H2O"]["K"]
            st.session_state['f_wqst'] = ref["H2O"]["Qst"]
            st.success("Template Loaded!")

        # EXPORT CSV BUTTON (NEW)
        st.markdown("---")
        st.subheader("Export")
        
        # Convert nested dict to flat dataframe for CSV
        export_list = []
        for mat, data in st.session_state['mof_data'].items():
            row = {"Material": mat}
            # Flatten CO2
            for k, v in data["CO2"].items(): row[f"CO2_{k}"] = v
            # Flatten H2O
            for k, v in data["H2O"].items(): row[f"H2O_{k}"] = v
            export_list.append(row)
        
        df_export = pd.DataFrame(export_list)
        csv_data = df_export.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            label="📥 Download Database as CSV",
            data=csv_data,
            file_name="mof_database_v17.csv",
            mime="text/csv"
        )

    with col_form:
        st.subheader("Manual Input")
        name = st.text_input("Name", key="f_name", value="New_Material")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### CO2 Parameters")
            cq1 = st.number_input("q_m1 (mmol/g)", key="f_cq1", value=5.0, format="%.4f")
            cb1 = st.number_input("b1 (1/bar)", key="f_cb1", value=1.0, format="%.4f")
            cq2 = st.number_input("q_m2 (mmol/g)", key="f_cq2", value=1.0, format="%.4f")
            cb2 = st.number_input("b2 (1/bar)", key="f_cb2", value=0.1, format="%.4f")
            cqst = st.number_input("Qst (kJ/mol)", key="f_cqst", value=30.0, format="%.4f")
        with c2:
            st.markdown("### H2O Parameters")
            wq = st.number_input("q_m (mmol/g)", key="f_wq", value=10.0, format="%.4f")
            wc = st.number_input("C (GAB)", key="f_wc", value=5.0, format="%.4f")
            wk = st.number_input("K (GAB)", key="f_wk", value=0.8, format="%.4f")
            wqst = st.number_input("Qst (kJ/mol)", key="f_wqst", value=40.0, format="%.4f")
            
        if st.button("💾 Save Material", type="primary"):
            st.session_state['mof_data'][name] = {
                "H2O": {"q_m": wq, "C": wc, "K": wk, "Qst": wqst},
                "CO2": {"q_m1": cq1, "b1": cb1, "q_m2": cq2, "b2": cb2, "Qst": cqst},
                "cif_url": "" 
            }
            st.success(f"Saved {name}!")

# --- TAB 4: REC ENGINE (UPDATED - ALL PARAMS) ---
with tabs[3]:
    st.header("⚗️ Detailed Sorbent Comparison")
    
    res = []
    T_std = 298
    Psat_298 = get_water_saturation_pressure(T_std)
    
    for n, d in st.session_state['mof_data'].items():
        c, w = d["CO2"], d["H2O"]
        cap_co2 = dual_site_langmuir(0.0004, c["q_m1"], c["b1"], c["q_m2"], c["b2"])
        cap_h2o = gab_model(Psat_298*0.5, w["q_m"], w["C"], w["K"], Psat_298)
        
        res.append({
            "Name": n,
            "CO2 (0.4mb)": f"{cap_co2:.4f}",
            "H2O (50%RH)": f"{cap_h2o:.4f}",
            # CO2 Params
            "CO2_qm1": f"{c['q_m1']:.4f}",
            "CO2_b1": f"{c['b1']:.4f}",
            "CO2_qm2": f"{c['q_m2']:.4f}",
            "CO2_b2": f"{c['b2']:.4f}",
            "CO2_Qst": f"{c['Qst']:.4f}",
            # H2O Params
            "H2O_qm": f"{w['q_m']:.4f}",
            "H2O_C": f"{w['C']:.4f}",
            "H2O_K": f"{w['K']:.4f}",
            "H2O_Qst": f"{w['Qst']:.4f}"
        })
    st.dataframe(pd.DataFrame(res))

# --- TAB 5: ISOTHERMS ---
with tabs[4]:
    st.header("📈 Equilibrium Isotherms (25°C)")
    col_c, col_w = st.columns(2)
    with col_c:
        st.subheader("CO2 Isotherms")
        fig_c = go.Figure()
        x_log = np.logspace(-4, 0, 100) 
        for n, d in st.session_state['mof_data'].items():
            cp = d["CO2"]
            y = dual_site_langmuir(x_log, cp["q_m1"], cp["b1"], cp["q_m2"], cp["b2"])
            fig_c.add_trace(go.Scatter(x=x_log * 1000, y=y, name=n)) 
        fig_c.update_xaxes(type="log", title="Pressure (mbar)")
        fig_c.update_yaxes(title="Loading (mmol/g)", range=[0, 10]) 
        st.plotly_chart(fig_c, use_container_width=True)

    with col_w:
        st.subheader("Water Isotherms")
        fig_w = go.Figure()
        Psat = get_water_saturation_pressure(298)
        x_lin_bar = np.linspace(0, Psat*0.99, 100)
        x_rh = (x_lin_bar / Psat) * 100
        for n, d in st.session_state['mof_data'].items():
            wp = d["H2O"]
            y = gab_model(x_lin_bar, wp["q_m"], wp["C"], wp["K"], Psat)
            fig_w.add_trace(go.Scatter(x=x_rh, y=y, name=n))
        fig_w.update_xaxes(title="Relative Humidity (%)")
        fig_w.update_yaxes(title="Loading (mmol/g)", range=[0, 100])
        st.plotly_chart(fig_w, use_container_width=True)

# --- TAB 6 & 7 (Standard) ---
with tabs[5]:
    st.header("3D Structure")
    if VIZ_AVAILABLE:
        sel = st.selectbox("Select", list(st.session_state['mof_data'].keys()), key='viz_sel')
        url = st.session_state['mof_data'][sel].get("cif_url")
        if url:
            view = py3Dmol.view(query=url, width=500, height=400)
            view.setStyle({'stick': {'radius': 0.15}, 'sphere': {'scale': 0.25}})
            view.zoomTo()
            showmol(view, height=400, width=500)
    else:
        st.warning("Install stmol/py3Dmol to view.")

with tabs[6]:
    st.header("🔄 Kinetic Cycler")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.subheader("Params")
        sel_k = st.selectbox("Material", list(st.session_state['mof_data'].keys()), key='kin_sel')
        n_cyc = st.number_input("Cycles", 1, 20, 3)
        ka = st.slider("Rate k_ads (min⁻¹)", 0.01, 2.0, 0.5)
        ta = st.number_input("Time (min)", 1, 100, 30, key='ta')
        Pa = st.number_input("P (bar)", 0.00001, 10.0, 1.0, format="%.4f")
        Ta = st.number_input("T (K)", 200, 400, 298, key='Ta')
        kd = st.slider("Rate k_des (min⁻¹)", 0.01, 2.0, 0.8)
        td = st.number_input("Time (min)", 1, 100, 30, key='td')
        Pd = st.number_input("P (bar)", 0.0, 10.0, 0.1, key='Pd', format="%.4f")
        Td = st.number_input("T (K)", 200, 500, 350, key='Td')
    with c2:
        p = st.session_state['mof_data'][sel_k]["CO2"]
        b1_a = adjust_affinity(p["b1"], Ta, 298, p["Qst"])
        b2_a = adjust_affinity(p["b2"], Ta, 298, p["Qst"])
        q_star_a = dual_site_langmuir(Pa, p["q_m1"], b1_a, p["q_m2"], b2_a)
        b1_d = adjust_affinity(p["b1"], Td, 298, p["Qst"])
        b2_d = adjust_affinity(p["b2"], Td, 298, p["Qst"])
        q_star_d = dual_site_langmuir(Pd, p["q_m1"], b1_d, p["q_m2"], b2_d)
        t_arr, q_arr = [], []
        curr_q, curr_t = 0.0, 0.0
        for i in range(n_cyc):
            t = np.linspace(0, ta, 30)
            q = simulate_kinetic_step(curr_q, q_star_a, ka, t)
            t_arr.extend(t + curr_t)
            q_arr.extend(q)
            curr_q, curr_t = q[-1], curr_t + ta
            t = np.linspace(0, td, 30)
            q = simulate_kinetic_step(curr_q, q_star_d, kd, t)
            t_arr.extend(t + curr_t)
            q_arr.extend(q)
            curr_q, curr_t = q[-1], curr_t + td
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_arr, y=q_arr, mode='lines', line=dict(width=2)))
        fig.add_hline(y=q_star_a, line_dash="dot")
        fig.add_hline(y=q_star_d, line_dash="dot")
        fig.update_layout(title="Kinetic Profile", xaxis_title="Time (min)", yaxis_title="Loading (mmol/g)")
        st.plotly_chart(fig, use_container_width=True)
