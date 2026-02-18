import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

st.set_page_config(page_title="AXIS Reserving Calculator", layout="wide")

FILE = Path("AXIS-3.xlsx")  # keep AXIS-3.xlsx in SAME folder as app.py
MONTHS_ALLOWED = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120]

ELR_MAP_DEFAULT = {"2008-2015": 0.65, "2016-2017": 0.75}
TAIL_MAP_DEFAULT = {"Optimistic": 1.05, "Baseline": 1.07, "Pessimistic": 1.09}


# DATA LOAD (cached)

@st.cache_data(show_spinner=False)
def load_raw_excel(file_path: str) -> pd.DataFrame:
    return pd.read_excel(file_path, sheet_name="Sheet1", header=None)

if not FILE.exists():
    st.error("AXIS-3.xlsx not found. Put AXIS-3.xlsx in the same folder as app.py.")
    st.stop()

df_raw = load_raw_excel(str(FILE))


# EXTRACTION HELPERS

def find_triangle_row(text: str) -> int:
    target = text.strip().lower()
    for r in range(df_raw.shape[0]):
        v0 = df_raw.iat[r, 0]
        v1 = df_raw.iat[r, 1]
        ok_label = isinstance(v0, str) and v0.strip().lower() == target
        ok_month = (v1 == 12) or (isinstance(v1, str) and v1.strip() == "12")
        if ok_label and ok_month:
            return r
    raise ValueError(f"Triangle header '{text}' not found.")

def list_triangle_anchors() -> list[str]:
    """Auto-discover triangle headers: col0 is string label AND col1 is 12."""
    anchors = []
    for r in range(df_raw.shape[0]):
        v0 = df_raw.iat[r, 0]
        v1 = df_raw.iat[r, 1]
        if isinstance(v0, str):
            ok_month = (v1 == 12) or (isinstance(v1, str) and str(v1).strip() == "12")
            if ok_month:
                label = v0.strip()
                if label and label not in anchors:
                    anchors.append(label)
    return anchors

def find_row_anywhere(text: str) -> int:
    target = text.strip().lower()
    for r in range(df_raw.shape[0]):
        for c in range(df_raw.shape[1]):
            v = df_raw.iat[r, c]
            if isinstance(v, str) and target in v.strip().lower():
                return r
    raise ValueError(f"Text '{text}' not found in sheet.")

def extract_summary() -> pd.DataFrame:
    headers = df_raw.iloc[7, 0:9].tolist()
    data = df_raw.iloc[8:20, 0:9].copy()
    data.columns = headers

    data = data[~data["Accident Year"].isna()].copy()
    data["Accident Year"] = data["Accident Year"].astype(str)

    for col in data.columns:
        if col == "Accident Year":
            continue
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data.set_index("Accident Year")

def extract_elr_assumptions() -> dict:
    r = find_row_anywhere("Expected loss ratio assumptions")

    g1 = str(df_raw.iat[r + 1, 1]).strip().replace("–", "-")
    v1 = float(df_raw.iat[r + 1, 2])

    g2 = str(df_raw.iat[r + 2, 1]).strip().replace("–", "-")
    v2 = float(df_raw.iat[r + 2, 2])

    return {g1: v1, g2: v2}

def extract_triangle(anchor: str) -> pd.DataFrame:
    r = find_triangle_row(anchor)
    months_raw = df_raw.iloc[r, 1:].tolist()

    months = []
    month_cols = []
    for i, m in enumerate(months_raw, start=1):
        if pd.isna(m):
            break
        try:
            mi = int(float(m))
        except Exception:
            break
        if mi in MONTHS_ALLOWED:
            months.append(str(mi))
            month_cols.append(i)
        else:
            break

    if len(months) < 2:
        raise ValueError(f"Could not read month columns for '{anchor}'.")

    data_rows, ay_list = [], []
    rr = r + 1
    while rr < df_raw.shape[0]:
        ay = df_raw.iat[rr, 0]
        if pd.isna(ay):
            break
        try:
            ay_str = str(int(float(ay)))
        except Exception:
            break
        vals = [df_raw.iat[rr, c] for c in month_cols]
        data_rows.append(vals)
        ay_list.append(ay_str)
        rr += 1

    tri = pd.DataFrame(data_rows, index=ay_list, columns=months)
    return tri.apply(pd.to_numeric, errors="coerce")


# DEVELOPMENT HELPERS

def make_age_to_age_triangle(cum_tri: pd.DataFrame) -> pd.DataFrame:
    cols = list(cum_tri.columns)
    ata = pd.DataFrame(index=cum_tri.index)
    for i in range(len(cols) - 1):
        a, b = cols[i], cols[i + 1]
        ata[f"{a}->{b}"] = cum_tri[b] / cum_tri[a]
    return ata.replace([np.inf, -np.inf], np.nan)

def select_ata_arithmetic(ata: pd.DataFrame) -> pd.DataFrame:
    selected = ata.mean(axis=0, skipna=True)
    counts = ata.notna().sum(axis=0)
    result = pd.DataFrame({"Arithmetic Avg": selected, "N": counts})

    links = list(result.index)
    from_ages = [l.split("->")[0] for l in links]
    to_ages = [l.split("->")[1] for l in links]
    ages = from_ages + [to_ages[-1]]

    cdf = pd.Series(index=ages, dtype=float)
    cdf.iloc[-1] = 1.0
    for i in range(len(ages) - 2, -1, -1):
        link = f"{ages[i]}->{ages[i+1]}"
        cdf.iloc[i] = float(result.loc[link, "Arithmetic Avg"]) * float(cdf.iloc[i + 1])

    result["CDF to 120"] = [cdf[age] for age in from_ages]
    return result

def select_ata_geometric(ata: pd.DataFrame) -> pd.DataFrame:
    geo_vals, counts = {}, {}
    for link in ata.columns:
        s = ata[link].dropna().astype(float)
        s = s[s > 0]
        counts[link] = int(s.shape[0])
        geo_vals[link] = np.nan if counts[link] == 0 else float(np.exp(np.mean(np.log(s))))

    result = pd.DataFrame({"Geometric Avg": pd.Series(geo_vals), "N": pd.Series(counts)})

    links = list(result.index)
    from_ages = [l.split("->")[0] for l in links]
    to_ages = [l.split("->")[1] for l in links]
    ages = from_ages + [to_ages[-1]]

    cdf = pd.Series(index=ages, dtype=float)
    cdf.iloc[-1] = 1.0
    for i in range(len(ages) - 2, -1, -1):
        link = f"{ages[i]}->{ages[i+1]}"
        cdf.iloc[i] = float(result.loc[link, "Geometric Avg"]) * float(cdf.iloc[i + 1])

    result["CDF to 120"] = [cdf[age] for age in from_ages]
    return result

def select_ata_volume_weighted(cum_tri: pd.DataFrame) -> pd.DataFrame:
    ages = list(cum_tri.columns)
    links = [f"{ages[i]}->{ages[i+1]}" for i in range(len(ages) - 1)]

    vw_vals, counts = {}, {}
    for i, link in enumerate(links):
        a, b = ages[i], ages[i + 1]
        mask = cum_tri[a].notna() & cum_tri[b].notna() & (cum_tri[a] > 0)
        counts[link] = int(mask.sum())
        vw_vals[link] = np.nan if counts[link] == 0 else float(cum_tri.loc[mask, b].sum() / cum_tri.loc[mask, a].sum())

    result = pd.DataFrame({"Volume Weighted Avg": pd.Series(vw_vals), "N": pd.Series(counts)})

    cdf = pd.Series(index=ages, dtype=float)
    cdf.iloc[-1] = 1.0
    for i in range(len(ages) - 2, -1, -1):
        cdf.iloc[i] = float(result.loc[links[i], "Volume Weighted Avg"]) * float(cdf.iloc[i + 1])

    result["CDF to 120"] = [cdf[age] for age in ages[:-1]]
    return result

def build_cdf120_map(cum_tri: pd.DataFrame, avg_method: str) -> tuple[pd.Series, pd.DataFrame]:
    ages = list(cum_tri.columns)

    if avg_method == "Arithmetic":
        ata = make_age_to_age_triangle(cum_tri)
        selected_df = select_ata_arithmetic(ata)
        cdf120_map = pd.Series(index=ages, dtype=float)
        for link in selected_df.index:
            age = link.split("->")[0]
            cdf120_map.loc[age] = float(selected_df.loc[link, "CDF to 120"])
        cdf120_map.loc[ages[-1]] = 1.0
        return cdf120_map, selected_df

    if avg_method == "Geometric":
        ata = make_age_to_age_triangle(cum_tri)
        selected_df = select_ata_geometric(ata)
        cdf120_map = pd.Series(index=ages, dtype=float)
        for link in selected_df.index:
            age = link.split("->")[0]
            cdf120_map.loc[age] = float(selected_df.loc[link, "CDF to 120"])
        cdf120_map.loc[ages[-1]] = 1.0
        return cdf120_map, selected_df

    if avg_method == "Volume-weighted":
        selected_df = select_ata_volume_weighted(cum_tri)
        cdf120_map = pd.Series(index=ages, dtype=float)
        for link in selected_df.index:
            age = link.split("->")[0]
            cdf120_map.loc[age] = float(selected_df.loc[link, "CDF to 120"])
        cdf120_map.loc[ages[-1]] = 1.0
        return cdf120_map, selected_df

    raise ValueError("avg_method must be 'Arithmetic', 'Geometric', or 'Volume-weighted'")

def triangle_latest_and_age(cum_tri: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    latest_val = cum_tri.apply(lambda r: r.dropna().iloc[-1], axis=1)
    latest_age = cum_tri.apply(lambda r: r.dropna().index[-1], axis=1)
    return latest_val, latest_age


# ELR MAP → SERIES

def build_elr_series(summary_df: pd.DataFrame, elr_map: dict) -> pd.Series:
    ay_int = pd.to_numeric(summary_df.index, errors="coerce")
    elr = pd.Series(index=summary_df.index, dtype=float)
    for k, v in elr_map.items():
        a, b = k.replace("–", "-").split("-")
        a, b = int(a), int(b)
        mask = (ay_int >= a) & (ay_int <= b)
        elr.loc[mask] = float(v)
    return elr


# RESERVING METHODS (IBNR locked to Current Reported)

def chain_ladder(cum_tri: pd.DataFrame, avg_method: str, tail: float, current_reported: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:

    cdf120_map, selected_df = build_cdf120_map(cum_tri, avg_method)
    latest_triangle, latest_age = triangle_latest_and_age(cum_tri)

    f120 = latest_age.map(lambda a: float(cdf120_map.get(a, np.nan)))
    dev120 = latest_triangle * f120
    ultimate = dev120 * float(tail)

    idx = ultimate.index.intersection(current_reported.index)
    res = pd.DataFrame({
        "Current Reported (Summary)": current_reported.loc[idx],
        "Triangle Latest": latest_triangle.loc[idx],
        "Latest Age": latest_age.loc[idx],
        "Factor to 120": f120.loc[idx],
        "Developed to 120": dev120.loc[idx],
        "Tail (120->Ult)": float(tail),
        "Ultimate": ultimate.loc[idx],
    })
    res["IBNR"] = res["Ultimate"] - res["Current Reported (Summary)"]
    return res, selected_df

def elr_method(summary_df: pd.DataFrame, elr_map: dict, current_reported: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Ultimate (ELR) = Earned Premium * ELR
    IBNR (ELR)     = Ultimate (ELR) - Current Reported
    """
    elr_series = build_elr_series(summary_df, elr_map)
    earned = pd.to_numeric(summary_df["Earned Premium"], errors="coerce")

    idx = earned.index.intersection(elr_series.index).intersection(current_reported.index)
    ultimate = earned.loc[idx] * elr_series.loc[idx]
    res = pd.DataFrame({
        "Current Reported (Summary)": current_reported.loc[idx],
        "Earned Premium": earned.loc[idx],
        "ELR": elr_series.loc[idx],
        "Ultimate (ELR)": ultimate,
    })
    res["IBNR (ELR)"] = res["Ultimate (ELR)"] - res["Current Reported (Summary)"]
    return res, elr_series.loc[idx]

def bf_method(summary_df: pd.DataFrame, cum_tri: pd.DataFrame, avg_method: str, tail: float, elr_map: dict, current_reported: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    BF:
      expected_ultimate = EP * ELR
      fult = CDF(latest->120) * tail
      %unreported = 1 - 1/fult
      IBNR = expected_ultimate * %unreported
      Ultimate = Current Reported + IBNR  (per exhibit)
    """
    earned = pd.to_numeric(summary_df["Earned Premium"], errors="coerce")
    elr_series = build_elr_series(summary_df, elr_map)

    cdf120_map, selected_df = build_cdf120_map(cum_tri, avg_method)
    _, latest_age = triangle_latest_and_age(cum_tri)

    idx = earned.index.intersection(elr_series.index).intersection(current_reported.index).intersection(latest_age.index)
    earned = earned.loc[idx]
    elr_series = elr_series.loc[idx]
    cur_rep = current_reported.loc[idx]
    latest_age = latest_age.loc[idx]

    expected_ultimate = earned * elr_series
    f120 = latest_age.map(lambda a: float(cdf120_map.get(a, np.nan)))
    fult = f120 * float(tail)

    pct_unreported = 1 - (1 / fult)
    ibnr = expected_ultimate * pct_unreported
    ultimate = cur_rep + ibnr

    res = pd.DataFrame({
        "Current Reported (Summary)": cur_rep,
        "Earned Premium": earned,
        "ELR": elr_series,
        "Latest Age": latest_age,
        "fult (Latest->Ult)": fult,
        "% Unreported": pct_unreported,
        "IBNR (BF)": ibnr,
        "Ultimate (BF)": ultimate,
        "Expected Ultimate (EP*ELR)": expected_ultimate,
    })
    return res, selected_df


# UI HELPERS

def fmt_money(x) -> str:
    if pd.isna(x):
        return ""
    return f"{x:,.0f}"

def display_round_2(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DISPLAY-ONLY copy rounded to 2 decimals (numeric cols only)."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(2)
    return out

def fmt_factor(x) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.6f}"


# STREAMLIT APP

st.title("AXIS Reserving Calculator")
st.caption(
    "Select a loss triangle and reserving method. Tail scenarios use fixed values (1.05 / 1.07 / 1.09). "
)

summary = extract_summary()
if summary.empty:
    st.error("Summary block could not be extracted or is empty. Check the Excel layout / extraction indices.")
    st.stop()

# Lock Current Reported
if "Reported Losses" not in summary.columns:
    st.error("Summary table must contain a 'Reported Losses' column to compute IBNR per exhibit.")
    st.stop()

current_reported = pd.to_numeric(summary["Reported Losses"], errors="coerce")
if current_reported.isna().all():
    st.error("'Reported Losses' exists but has no numeric values.")
    st.stop()

triangle_options = list_triangle_anchors()
if not triangle_options:
    st.error("Could not find any triangle headers (label in col A + '12' in col B). Check Excel layout.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Controls")

    default_tri_idx = triangle_options.index("Reported Losses") if "Reported Losses" in triangle_options else 0
    triangle_choice = st.selectbox("Loss triangle", triangle_options, index=default_tri_idx)

    technique = st.selectbox("Reserving technique", ["Chain Ladder", "ELR", "Bornhuetter-Ferguson"], index=0)

    # Averaging needed for CL and BF
    if technique in ("Chain Ladder", "Bornhuetter-Ferguson"):
        avg_method = st.selectbox("Age-to-age averaging", ["Arithmetic", "Geometric", "Volume-weighted"], index=0)
    else:
        avg_method = None

    # Tail settings for CL and BF (fixed scenarios + custom override)
    if technique in ("Chain Ladder", "Bornhuetter-Ferguson"):
        st.subheader("Tail factor")
        tail_choice = st.selectbox("Tail scenario", list(TAIL_MAP_DEFAULT.keys()), index=1)
        use_custom_tail = st.checkbox("Custom tail value", value=False)
        tail_value = float(TAIL_MAP_DEFAULT[tail_choice])
        if use_custom_tail:
            tail_value = st.number_input(
                "Tail (120→Ultimate)",
                value=float(tail_value),
                min_value=1.0,
                step=0.01,
                format="%.6f",
            )
    else:
        tail_choice, tail_value = None, None

    # ELR assumptions (back to simple UI)
    st.subheader("Expected loss ratios")
    use_custom_elr = st.checkbox("Custom ELR values", value=False)

    try:
        axis_elr_from_file = extract_elr_assumptions()
    except Exception:
        axis_elr_from_file = ELR_MAP_DEFAULT.copy()

    if use_custom_elr:
        elr_0815 = st.number_input(
            "ELR for 2008–2015",
            value=float(axis_elr_from_file.get("2008-2015", 0.65)),
            min_value=0.0,
            max_value=2.0,
            step=0.01,
            format="%.4f",
        )
        elr_1617 = st.number_input(
            "ELR for 2016–2017",
            value=float(axis_elr_from_file.get("2016-2017", 0.75)),
            min_value=0.0,
            max_value=2.0,
            step=0.01,
            format="%.4f",
        )
        ELR_MAP = {"2008-2015": float(elr_0815), "2016-2017": float(elr_1617)}
    else:
        ELR_MAP = axis_elr_from_file.copy()

    st.divider()
    show_triangles = st.checkbox("Show triangles / derivations", value=False)

    st.subheader("Downloads")
    with open(FILE, "rb") as f:
        st.download_button(
            label="Download AXIS-3.xlsx",
            data=f.read(),
            file_name="AXIS-3.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

# Load selected triangle
try:
    cum_tri = extract_triangle(triangle_choice)
except Exception as e:
    st.error(f"Could not extract triangle '{triangle_choice}': {e}")
    st.stop()

if cum_tri.isna().all().all():
    st.error(f"Triangle '{triangle_choice}' extracted but all values are NaN. Check Excel layout / anchor row.")
    st.stop()

# Layout
colA, colB = st.columns([1.15, 1])

with colA:
    st.subheader("Inputs snapshot")
    st.write(f"**Triangle:** {triangle_choice}")
    st.write("**IBNR Definition:** Ultimate − Current Reported")
    st.write(f"**Technique:** {technique}")
    if avg_method:
        st.write(f"**Averaging:** {avg_method}")
    if tail_value is not None:
        st.write(f"**Tail (120→Ult):** {tail_value:.6f} ({tail_choice})")
    st.write(f"**ELR map:** {ELR_MAP}")

with colB:
    st.subheader("Summary preview (rounded for display)")
    summary_display = summary.copy()
    for c in summary_display.columns:
        if pd.api.types.is_numeric_dtype(summary_display[c]):
            summary_display[c] = summary_display[c].round(2)
    st.dataframe(summary_display, use_container_width=True)

# Derivations expander (kept)
# Triangle display
if show_triangles:
    with st.expander("Triangle(s)", expanded=True):
        st.write(f"**{triangle_choice} cumulative triangle**")
        st.dataframe(display_round_2(cum_tri), use_container_width=True)

        if technique in ("Chain Ladder", "Bornhuetter-Ferguson"):
            ata = make_age_to_age_triangle(cum_tri)
            st.write("**Age-to-age (link ratio) triangle**")
            st.dataframe(display_round_2(ata), use_container_width=True)

st.divider()

# CALCULATE

selected_table = None
results = None

try:
    if technique == "Chain Ladder":
        results, selected_table = chain_ladder(cum_tri, avg_method, tail_value, current_reported)

        st.subheader("Selected age-to-age factors & CDF to 120")
        st.dataframe(
            selected_table.style.format({c: fmt_factor for c in selected_table.columns if c != "N"}),
            use_container_width=True,
        )

        st.subheader("Chain Ladder results")
        st.dataframe(
            results.style.format({
                "Current Reported (Summary)": fmt_money,
                "Triangle Latest": fmt_money,
                "Factor to 120": fmt_factor,
                "Developed to 120": fmt_money,
                "Tail (120->Ult)": fmt_factor,
                "Ultimate": fmt_money,
                "IBNR": fmt_money,
            }),
            use_container_width=True,
        )
        st.metric("Total IBNR (CL)", fmt_money(results["IBNR"].sum()))
        st.metric("Total Ultimate (CL)", fmt_money(results["Ultimate"].sum()))

    elif technique == "ELR":
        results, elr_series = elr_method(summary, ELR_MAP, current_reported)

        st.subheader("ELR results")
        st.dataframe(
            results.style.format({
                "Current Reported (Summary)": fmt_money,
                "Earned Premium": fmt_money,
                "ELR": fmt_factor,
                "Ultimate (ELR)": fmt_money,
                "IBNR (ELR)": fmt_money,
            }),
            use_container_width=True,
        )
        st.metric("Total IBNR (ELR)", fmt_money(results["IBNR (ELR)"].sum()))
        st.metric("Total Ultimate (ELR)", fmt_money(results["Ultimate (ELR)"].sum()))

        neg = int((results["IBNR (ELR)"] < 0).sum())
        if neg > 0:
            st.warning(f"{neg} accident years have negative IBNR under ELR (current reported exceeds ELR ultimate).")

    else:  # Bornhuetter-Ferguson
        results, selected_table = bf_method(summary, cum_tri, avg_method, tail_value, ELR_MAP, current_reported)

        st.subheader("Selected age-to-age factors & CDF to 120 (used to compute fult)")
        st.dataframe(
            selected_table.style.format({c: fmt_factor for c in selected_table.columns if c != "N"}),
            use_container_width=True,
        )

        st.subheader("Bornhuetter–Ferguson results")
        st.dataframe(
            results.style.format({
                "Current Reported (Summary)": fmt_money,
                "Earned Premium": fmt_money,
                "ELR": fmt_factor,
                "fult (Latest->Ult)": fmt_factor,
                "% Unreported": fmt_factor,
                "IBNR (BF)": fmt_money,
                "Ultimate (BF)": fmt_money,
                "Expected Ultimate (EP*ELR)": fmt_money,
            }),
            use_container_width=True,
        )
        st.metric("Total IBNR (BF)", fmt_money(results["IBNR (BF)"].sum()))
        st.metric("Total Ultimate (BF)", fmt_money(results["Ultimate (BF)"].sum()))

except Exception as e:
    st.error(f"Calculation failed: {e}")
    st.stop()


# DOWNLOADS

st.divider()
st.subheader("Download outputs")
dl_cols = st.columns(3)

with dl_cols[0]:
    if results is not None:
        st.download_button(
            "Download results (CSV)",
            data=results.to_csv().encode("utf-8"),
            file_name=f"results_{technique.replace(' ', '_').lower()}_{triangle_choice.replace(' ', '_').lower()}.csv",
            mime="text/csv",
        )

with dl_cols[1]:
    if selected_table is not None:
        st.download_button(
            "Download selected factors (CSV)",
            data=selected_table.to_csv().encode("utf-8"),
            file_name=f"selected_factors_{avg_method.lower()}_{triangle_choice.replace(' ', '_').lower()}.csv",
            mime="text/csv",
        )

with dl_cols[2]:
    st.download_button(
        "Download summary table (CSV)",
        data=summary.to_csv().encode("utf-8"),
        file_name="axis_summary.csv",
        mime="text/csv",
    )

st.caption(
    "Notes: (1) Tail scenarios are fixed at 1.05 / 1.07 / 1.09 (custom override available). "
    "(2) ELR assumptions follow AXIS groupings (2008–2015, 2016–2017) with optional user override. "
)
