import base64, math, os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import core

st.set_page_config(page_title="GeneExplainer", layout="wide")
st.title("GeneExplainer")
st.session_state.setdefault("up", [])
st.session_state.setdefault("down", [])

tab_home, tab_an, tab_help, tab_about = st.tabs(["Home", "Analyze", "Help", "About"])

# ---------------- Home ----------------
def helix_svg(w=360, h=220):
    pts1, pts2, rungs = [], [], []
    n = 60
    for i in range(n + 1):
        x = 10 + i * (w - 20) / n
        y1 = h / 2 + 70 * math.sin(i / 6.0)
        y2 = h / 2 - 70 * math.sin(i / 6.0)
        pts1.append(f"{x:.1f},{y1:.1f}"); pts2.append(f"{x:.1f},{y2:.1f}")
        if i % 3 == 0:
            rungs.append(f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" stroke="#7fd6e6" stroke-opacity="0.35" stroke-width="2"/>')
    svg = (f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg">'
           + "".join(rungs)
           + f'<polyline points="{" ".join(pts1)}" fill="none" stroke="#ff4b4b" stroke-width="4" stroke-linecap="round"/>'
           + f'<polyline points="{" ".join(pts2)}" fill="none" stroke="#4fc3d9" stroke-width="4" stroke-linecap="round"/></svg>')
    b64 = base64.b64encode(svg.encode()).decode()
    return f'<img src="data:image/svg+xml;base64,{b64}" style="width:100%;max-width:420px"/>'


def card(num, title, text, color):
    return (f'<div style="background:#1b1e27;border:1px solid #2c303c;border-top:4px solid {color};border-radius:12px;'
            f'padding:18px;height:100%"><div style="color:{color};font-size:30px;font-weight:700">{num}</div>'
            f'<div style="font-size:18px;font-weight:600;margin:4px 0 6px">{title}</div>'
            f'<div style="color:#b8bcc8;font-size:14px;line-height:1.5">{text}</div></div>')


def tile(big, small):
    return (f'<div style="background:#1b1e27;border:1px solid #2c303c;border-radius:12px;padding:14px;text-align:center">'
            f'<div style="font-size:22px;font-weight:700;color:#4fc3d9">{big}</div>'
            f'<div style="color:#b8bcc8;font-size:13px">{small}</div></div>')


def flow_box(label, color):
    return (f'<div style="background:{color}22;border:2px solid {color};border-radius:10px;padding:10px 14px;'
            f'text-align:center;font-size:14px;font-weight:600;min-width:120px">{label}</div>')


with tab_home:
    h1, h2 = st.columns([3, 2])
    with h1:
        st.markdown('<div style="font-size:15px;color:#ff4b4b;font-weight:600;letter-spacing:2px;margin-top:20px">'
                    'GENAI FOR TRANSCRIPTOMICS</div>', unsafe_allow_html=True)
        st.header("From gene list to biological story")
        st.write("GeneExplainer takes the genes that are up- or down-regulated in a differential expression "
                 "analysis, finds the GO and KEGG pathways they are enriched in, and uses a generative AI model "
                 "to explain the result in plain language. The AI may only use the enrichment results it is given, "
                 "and a grounding check scores how well it follows that rule.")
        st.markdown("**Case study:** lung tumour vs normal tissue (GEO: GSE10072)")
    with h2:
        st.markdown(helix_svg(), unsafe_allow_html=True)

    st.markdown("### How it works")
    c1, c2, c3 = st.columns(3)
    c1.markdown(card("1", "Input", "Upload a limma results CSV with gene, logFC and adj.P.Val, or paste your own "
                     "up- and down-regulated gene lists.", "#ff4b4b"), unsafe_allow_html=True)
    c2.markdown(card("2", "Enrichment", "Genes are tested against GO Biological Process and KEGG pathways "
                     "using Enrichr, separately for up and down.", "#f5a623"), unsafe_allow_html=True)
    c3.markdown(card("3", "GenAI explanation", "Gemini writes a pathway-level story from those results only, "
                     "citing each pathway by name.", "#4fc3d9"), unsafe_allow_html=True)

    st.markdown("### Pipeline")
    arrow = '<div style="font-size:22px;color:#7a7f8f">&#10132;</div>'
    flow = ('<div style="display:flex;align-items:center;justify-content:center;gap:10px;flex-wrap:wrap;margin:6px 0 18px">'
            + flow_box("GEO data<br>GSE10072", "#ff4b4b") + arrow + flow_box("limma<br>DE genes", "#f5a623") + arrow
            + flow_box("Enrichr<br>GO + KEGG", "#7bd88f") + arrow + flow_box("Gemini<br>explanation", "#4fc3d9") + arrow
            + flow_box("Grounding<br>check", "#b48ead") + "</div>")
    st.markdown(flow, unsafe_allow_html=True)

    st.markdown("### Why it is trustworthy")
    t1, t2, t3, t4 = st.columns(4)
    t1.markdown(tile("2", "pathway databases"), unsafe_allow_html=True)
    t2.markdown(tile("3", "grounding metrics"), unsafe_allow_html=True)
    t3.markdown(tile("vs baseline", "GenAI compared with a template"), unsafe_allow_html=True)
    t4.markdown(tile("Cached", "repeat runs work offline"), unsafe_allow_html=True)
    st.write("")
    st.info("Ready? Open the **Analyze** tab, load your data and click **Start analysis**.")

# ---------------- Help ----------------
with tab_help:
    st.header("Input requirements")
    st.subheader("Option A: limma results CSV")
    st.write("One row per gene, with these exact column names:")
    st.code("gene,logFC,adj.P.Val\nTOP2A,2.31,0.0001\nAGER,-2.8,0.00002", language="text")
    st.write("The R script `prep_geo_de.R` creates this file from GSE10072. Genes with adj.P.Val < 0.05 and "
             "|logFC| >= the cutoff are used (top 300 per direction).")
    st.subheader("Option B: paste gene lists")
    st.write("Official gene symbols separated by spaces, commas or new lines. At least 5 genes per list.")
    st.subheader("API key")
    st.write("Get a free Gemini key at aistudio.google.com and paste it in the Analyze tab. "
             "Without a key the app shows a simple template explanation instead.")

# ---------------- About ----------------
with tab_about:
    st.header("About")
    st.write("Case study: explaining differential expression results in lung tumour vs normal tissue "
             "(GEO dataset GSE10072) with generative AI.")
    st.subheader("Technologies")
    st.markdown("- **Enrichr** for GO/KEGG enrichment\n- **Google Gemini** as the GenAI model\n"
                "- **Streamlit** for the interface\n- **limma (R)** for differential expression")
    st.subheader("Team")
    st.write("Group members: ____________________")
    st.caption("Interface layout inspired by StreamDEA (Mohamed Sinan M).")

# ---------------- Analyze ----------------
def show_enrichment(up_t, down_t):
    cols = st.columns(2)
    for col, name, t in [(cols[0], "Up-regulated", up_t), (cols[1], "Down-regulated", down_t)]:
        with col:
            st.markdown(f"**{name}**")
            if t.empty:
                st.write("No significant terms.")
                continue
            fig, ax = plt.subplots(figsize=(5, 3.2))
            ax.barh(t["term"][::-1], [-math.log10(p) for p in t["adj_p"]][::-1])
            ax.set_xlabel("-log10 adjusted p"); ax.tick_params(axis="y", labelsize=7)
            plt.tight_layout(); st.pyplot(fig)
            st.dataframe(t.assign(overlap=t["overlap"].apply(", ".join))[["library", "term", "adj_p", "overlap"]])


def run(up, down, api_key, model, context, n_terms):
    try:
        with st.spinner("Running enrichment (Enrichr)..."):
            up_all = core.enrich(up, "up") if len(up) >= 5 else pd.DataFrame()
            down_all = core.enrich(down, "down") if len(down) >= 5 else pd.DataFrame()
    except Exception as e:
        st.error(f"Enrichment failed (check internet): {e}")
        return
    up_t, down_t = core.top_terms(up_all, n_terms), core.top_terms(down_all, n_terms)
    st.subheader("1. Enrichment results")
    show_enrichment(up_t, down_t)

    allowed = list(up_t["term"]) + list(down_t["term"])
    base = core.template_baseline(up_t, down_t)
    llm_text = None
    if api_key:
        try:
            with st.spinner("Asking Gemini..."):
                llm_text = core.call_gemini(core.build_prompt(context, up_t, down_t, len(up), len(down)),
                                            api_key, model)
        except Exception as e:
            st.warning(f"LLM call failed, showing template baseline only: {e}")
    else:
        st.warning("No API key: showing the template baseline only.")
    st.subheader("2. Explanation")
    st.markdown(llm_text if llm_text else base)

    st.subheader("3. Grounding check")
    genes = up + down
    rows = {}
    if llm_text:
        rows["GenAI (Gemini)"] = core.grounding_metrics(llm_text, allowed, genes, allowed)
    rows["Template baseline"] = core.grounding_metrics(base, allowed, genes, allowed)
    m = pd.DataFrame(rows).T
    st.dataframe(m[["term_precision", "term_coverage", "gene_precision", "n_terms_cited", "n_gene_mentions"]])
    if llm_text and rows["GenAI (Gemini)"]["unsupported_genes"]:
        st.caption("Gene-like words not in your input: " + ", ".join(rows["GenAI (Gemini)"]["unsupported_genes"]))
    st.caption("term_precision: cited pathways that exist in the enrichment table. term_coverage: share of top "
               "terms mentioned. gene_precision: gene-like words found in your input (heuristic).")


with tab_an:
    s1, s2 = st.columns(2)
    api_key = s1.text_input("Gemini API key", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    model = s2.text_input("Gemini model", value="gemini-3.8-flash")
    s3, s4, s5 = st.columns([2, 1, 1])
    context = s3.text_input("Study context", value="Lung tumour vs normal lung tissue (GEO: GSE10072)")
    lfc = s4.number_input("|logFC| cutoff", 0.0, 5.0, 1.0, 0.1)
    n_terms = s5.slider("Top terms", 3, 15, 8)

    st.write("Upload your gene results")
    f = st.file_uploader("CSV with columns: gene, logFC, adj.P.Val", type="csv")
    if st.button("Load demo data"):
        st.session_state["up"], st.session_state["down"] = core.DEMO_UP, core.DEMO_DOWN
        st.info("Demo loaded: an illustrative lung tumour vs normal list, not real GSE10072 output.")
    with st.expander("Or paste gene lists"):
        p1, p2 = st.columns(2)
        pu = core.parse_genes(p1.text_area("Up-regulated genes", height=120))
        pd_ = core.parse_genes(p2.text_area("Down-regulated genes", height=120))

    up, down = st.session_state["up"], st.session_state["down"]
    if f is not None:
        try:
            up, down = core.split_de_table(pd.read_csv(f), lfc=lfc)
        except KeyError:
            st.error("The CSV needs columns named: gene, logFC, adj.P.Val")
    if pu or pd_:
        up, down = pu, pd_
    st.write(f"**{len(up)}** up-regulated and **{len(down)}** down-regulated genes loaded.")

    if st.button("Start analysis", type="primary", disabled=not (len(up) >= 5 or len(down) >= 5)):
        run(up, down, api_key, model, context, n_terms)
