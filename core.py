"""Core logic for the Gene/GEO explainer: enrichment, prompting, LLM call, grounding metrics."""
import hashlib, json, os, re
import pandas as pd
import requests

ENRICHR = "https://maayanlab.cloud/Enrichr"
LIBS = ["GO_Biological_Process_2023", "KEGG_2021_Human"]
CACHE_DIR = ".cache"

# Illustrative lung tumour vs normal list (NOT computed from GSE10072; replace with your own results).
DEMO_UP = ("SPP1 COL1A1 COL1A2 COL3A1 COL11A1 MMP1 MMP11 MMP12 TOP2A MKI67 CDK1 CCNB1 CCNB2 UBE2C "
           "FOXM1 BIRC5 TYMS AURKA CEP55 PTTG1 CDC20 BUB1 PLK1 KIF11 KIF20A NUSAP1 RRM2 MCM2 MCM6 CENPF "
           "GREM1 SPINK1 CST1 FN1 THBS2 POSTN").split()
DEMO_DOWN = ("AGER SFTPC SFTPB SFTPA1 SFTPD NAPSA CAV1 CAV2 EMP2 SCGB1A1 FABP4 CLDN18 TEK EDNRB ADH1B "
             "SPARCL1 FHL1 TGFBR3 PECAM1 VWF CLEC3B AQP1 CAVIN2 HPGD RTKN2 GPM6A CLIC5 EPAS1 CDH5 "
             "KANK3 STX11 SELENBP1").split()

STOP = {"GO", "KEGG", "DNA", "RNA", "TNBC", "NOT", "THE", "AND", "FOR", "ARE", "WITH", "ECM", "NSCLC",
        "MRNA", "ATP", "GTP", "DE", "DEG", "DEGS", "PI3K", "AKT", "MAPK"}


def _key(*parts):
    return hashlib.sha256(json.dumps(parts, default=str).encode()).hexdigest()[:24]


def cache_get(k):
    p = os.path.join(CACHE_DIR, k + ".json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def cache_put(k, obj):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, k + ".json"), "w") as f:
        json.dump(obj, f)


def parse_genes(text):
    seen, out = set(), []
    for g in re.split(r"[\s,;]+", text.upper()):
        if g and g not in seen:
            seen.add(g); out.append(g)
    return out


def split_de_table(df, lfc=1.0, padj=0.05, top=300):
    """df needs columns: gene, logFC, adj.P.Val (limma topTable export)."""
    df = df.rename(columns={c: c.strip() for c in df.columns})
    d = df[(df["adj.P.Val"] < padj) & df["gene"].notna() & (df["gene"].astype(str) != "")].copy()
    d["gene"] = d["gene"].astype(str).str.split("///").str[0].str.strip().str.upper()
    d = d.sort_values("adj.P.Val").drop_duplicates("gene")
    up = d[d["logFC"] >= lfc]["gene"].head(top).tolist()
    down = d[d["logFC"] <= -lfc]["gene"].head(top).tolist()
    return up, down


def enrich(genes, label="list"):
    """Enrichr (GO BP + KEGG). Results are cached on disk so the demo survives bad wifi."""
    k = _key("enr", sorted(genes), LIBS)
    hit = cache_get(k)
    if hit is not None:
        return pd.DataFrame(hit)
    r = requests.post(f"{ENRICHR}/addList",
                      files={"list": (None, "\n".join(genes)), "description": (None, label)}, timeout=30)
    r.raise_for_status()
    uid = r.json()["userListId"]
    rows = []
    for lib in LIBS:
        j = requests.get(f"{ENRICHR}/enrich", params={"userListId": uid, "backgroundType": lib},
                         timeout=30).json()
        for x in j.get(lib, []):
            rows.append(dict(library=lib, term=re.sub(r"\s*\(GO:\d+\)", "", x[1]),
                             pvalue=x[2], adj_p=x[6], overlap=x[5]))
    cache_put(k, rows)
    return pd.DataFrame(rows)


def top_terms(df, n=8, alpha=0.05):
    if df.empty:
        return df
    d = df[df["adj_p"] < alpha].sort_values("adj_p")
    return d.head(n).reset_index(drop=True)


def build_prompt(context, up_terms, down_terms, n_up, n_down):
    def fmt(df):
        if df.empty:
            return "(no significant terms)"
        return "\n".join(f"- [{r.term}] adj.p={r.adj_p:.2e}; genes: {', '.join(r.overlap)}"
                         for r in df.itertuples())
    return f"""You are a careful bioinformatics assistant. Context: {context}
A differential expression analysis found {n_up} up-regulated and {n_down} down-regulated genes.
Below are the ONLY enrichment results you may use (GO Biological Process and KEGG, Enrichr).

UP-REGULATED enriched terms:
{fmt(up_terms)}

DOWN-REGULATED enriched terms:
{fmt(down_terms)}

Write a pathway-level explanation for a biologist in 3 short sections:
1. What is switched on, 2. What is switched off, 3. Overall biological story.
Rules:
- Cite every pathway by copying its name EXACTLY inside square brackets, e.g. [{'term name'}].
- Mention only genes that appear in the lists above; name 2-4 key genes per pathway.
- Do not invent pathways, genes, statistics or clinical claims. If evidence is weak, say so.
- End with one sentence on limitations (enrichment is associative, not causal)."""


FALLBACK_MODELS = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-2.5-flash"]
BASE = "https://generativelanguage.googleapis.com/v1beta"


def list_models(api_key):
    """Ask Google which models this key can use for text generation."""
    r = requests.get(f"{BASE}/models", headers={"x-goog-api-key": api_key}, params={"pageSize": 200}, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Google rejected the key when listing models ({r.status_code}): {r.text[:300]}")
    names = [m["name"].split("/")[-1] for m in r.json().get("models", [])
             if "generateContent" in m.get("supportedGenerationMethods", [])]
    flash = [n for n in names if "flash" in n and not any(x in n for x in ("tts", "live", "image", "audio", "lite", "transcribe", "omni", "robotics"))]
    return flash + [n for n in names if n not in flash]


def call_gemini(prompt, api_key, model="gemini-3.8-flash"):
    """Tries the chosen model, known fallbacks, then whatever models Google says this key can use."""
    k = _key("llm", prompt)
    hit = cache_get(k)
    if hit is not None:
        return hit["text"]
    tried, errors = set(), []

    def attempt(m):
        tried.add(m)
        r = requests.post(f"{BASE}/models/{m}:generateContent", headers={"x-goog-api-key": api_key},
                          json={"contents": [{"parts": [{"text": prompt}]}],
                                "generationConfig": {"temperature": 0.2}}, timeout=90)
        if r.status_code == 200:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        errors.append(f"{m}: {r.status_code} {r.text[:200]}")
        return None

    for m in [model] + FALLBACK_MODELS:
        if m in tried:
            continue
        text = attempt(m)
        if text:
            cache_put(k, {"text": text}); return text
    try:
        for m in list_models(api_key)[:4]:
            if m in tried:
                continue
            text = attempt(m)
            if text:
                cache_put(k, {"text": text}); return text
    except Exception as e:
        errors.append(str(e))
    raise RuntimeError("All Gemini attempts failed. Details: " + " | ".join(errors[-4:]))


def template_baseline(up_terms, down_terms):
    """Non-LLM baseline: fixed template filled with the top terms. Used for comparison and offline fallback."""
    def line(df, word):
        if df.empty:
            return f"No significant pathways were found for {word}-regulated genes."
        names = "; ".join(f"[{t}]" for t in df["term"].head(5))
        return f"Genes {word}-regulated are enriched in: {names}."
    return (line(up_terms, "up") + "\n\n" + line(down_terms, "down") +
            "\n\nEnrichment is associative, not causal.")


def grounding_metrics(text, allowed_terms, allowed_genes, top_n_terms):
    """Deterministic checks of how well an explanation sticks to the evidence it was given."""
    allowed_l = {t.lower() for t in allowed_terms}
    cited = [c.strip() for c in re.findall(r"\[([^\]]+)\]", text)]
    term_precision = (sum(c.lower() in allowed_l for c in cited) / len(cited)) if cited else 0.0
    cited_l = {c.lower() for c in cited}
    coverage = (sum(t.lower() in cited_l for t in top_n_terms) / len(top_n_terms)) if top_n_terms else 0.0
    cand = {t for t in re.findall(r"\b[A-Z][A-Z0-9]{2,9}\b", re.sub(r"\[[^\]]+\]", " ", text))} - STOP
    genes_set = {g.upper() for g in allowed_genes}
    gene_precision = (len(cand & genes_set) / len(cand)) if cand else 1.0
    return {"term_precision": term_precision, "term_coverage": coverage,
            "gene_precision": gene_precision, "n_terms_cited": len(cited),
            "n_gene_mentions": len(cand), "unsupported_genes": sorted(cand - genes_set)}
