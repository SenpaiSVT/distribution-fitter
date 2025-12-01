import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import stats

# ---------- PAGE CONFIG ---------- #

st.set_page_config(
    page_title="Data Distribution Fitter",
    layout="wide"
)

# ---------- THEME DEFINITIONS ---------- #

THEMES = {
    "Light (Blue & Red)": {
        "bg_color": "#f6f6f9",
        "text_color": "#222222",
        "ax_facecolor": "#ffffff",
        "hist_edgecolor": "white",
        "hist_alpha": 0.55,
        "hist_color": "#7fa6e5",
        "line_color_main": "#d85f5f",
        "line_color_shadow": "#b03636",
        "grid": True,
        "dark": False,
        "use_viridis": False,
        "neon": False,
    },
    "Dark (Teal & Orange)": {
        "bg_color": "#050816",
        "text_color": "#f5f5f5",
        "ax_facecolor": "#111827",
        "hist_edgecolor": "#111827",
        "hist_alpha": 0.8,      # a bit stronger
        "hist_color": "#14b8a6",
        "line_color_main": "#fb923c",
        "line_color_shadow": "#ea580c",
        "grid": True,
        "dark": True,
        "use_viridis": False,
        "neon": False,
    },
    "Viridis gradient": {
        "bg_color": "#0b1020",
        "text_color": "#f5f5f5",
        "ax_facecolor": "#020617",
        "hist_edgecolor": "none",
        "hist_alpha": 1.0,      # solid bars, easier to see
        "hist_color": None,     # not used; we use cmap instead
        "line_color_main": "#facc15",
        "line_color_shadow": "#a16207",
        "grid": False,
        "dark": True,
        "use_viridis": True,
        "neon": False,
    },
    "Soft Seaborn-like": {
        "bg_color": "#f3f4f6",
        "text_color": "#111827",
        "ax_facecolor": "#ffffff",
        "hist_edgecolor": "#e5e7eb",
        "hist_alpha": 0.7,
        "hist_color": "#60a5fa",
        "line_color_main": "#ec4899",
        "line_color_shadow": "#be185d",
        "grid": True,
        "dark": False,
        "use_viridis": False,
        "neon": False,
    },
    "Neon (Cyber)": {  # new theme
        "bg_color": "#020617",
        "text_color": "#f9fafb",
        "ax_facecolor": "#020617",
        "hist_edgecolor": "#22c55e",
        "hist_alpha": 1.0,       # full to show heights clearly
        "hist_color": "#22c55e",  # neon teal
        "line_color_main": "#ff7bff",   # bright magenta
        "line_color_shadow": "#fb37ff", # glow behind
        "grid": True,
        "dark": True,
        "use_viridis": False,
        "neon": True,
    },
}

# ---------- SIDEBAR: GLOBAL THEME SELECTION ---------- #

st.sidebar.title("Settings")

theme_name = st.sidebar.selectbox(
    "Color / style theme",
    list(THEMES.keys()),
    index=0
)
THEME = THEMES[theme_name]

# ---------- BASIC MATPLOTLIB STYLE (FONT, GRID SIZES) ---------- #

plt.style.use("default")
plt.rcParams.update({
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#333333" if not THEME["dark"] else "#e5e7eb",
    "axes.labelcolor": THEME["text_color"],
    "xtick.color": THEME["text_color"],
    "ytick.color": THEME["text_color"],
    "text.color": THEME["text_color"],
    "axes.grid": THEME["grid"],
    "grid.alpha": 0.25 if not THEME.get("neon", False) else 0.35,
    "grid.linestyle": "--" if not THEME.get("neon", False) else ":",
})

# (no global CSS here – we let Streamlit handle the page)


# ---------- DISTRIBUTIONS ---------- #

DISTRIBUTIONS = {
    "Normal (Gaussian)": stats.norm,
    "Gamma": stats.gamma,
    "Exponential": stats.expon,
    "Lognormal": stats.lognorm,
    "Weibull (min)": stats.weibull_min,
    "Chi-square": stats.chi2,
    "Beta": stats.beta,
    "Uniform": stats.uniform,
    "Cauchy": stats.cauchy,
    "Student t": stats.t,
}

# ---------- HELPER FUNCTIONS ---------- #

def parse_manual_data(text: str) -> np.ndarray:
    """Turn a free-form string into a 1D numpy array of floats."""
    if not text.strip():
        return np.array([])
    pieces = text.replace(",", " ").split()
    numbers = []
    for p in pieces:
        try:
            numbers.append(float(p))
        except ValueError:
            pass
    return np.array(numbers)


def compute_fit_error(data: np.ndarray, dist, params):
    """Compare histogram density to PDF values and return MSE and MAE."""
    data = np.asarray(data)
    # use same binning logic as our plots
    n_bins = max(5, min(30, max(1, len(data) // 2)))
    counts, bin_edges = np.histogram(data, bins=n_bins, density=True)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    x_grid = np.linspace(data.min(), data.max(), 400)
    pdf_grid = dist.pdf(x_grid, *params)
    pdf_at_centers = np.interp(bin_centers, x_grid, pdf_grid)

    mse = np.mean((counts - pdf_at_centers) ** 2)
    mae = np.mean(np.abs(counts - pdf_at_centers))
    return mse, mae


def split_params(params):
    """
    Split scipy's parameters into:
    - shape_params: list
    - center: renamed from loc
    - spread: renamed from scale
    """
    if len(params) <= 2:
        shape_params = []
        center = params[0]
        spread = params[1]
    else:
        shape_params = list(params[:-2])
        center = params[-2]
        spread = params[-1]
    return shape_params, center, spread


def make_histogram_with_pdf(data: np.ndarray, dist, params, title: str):
    """
    Return a themed matplotlib Figure with histogram + PDF curve.

    - Works the same for all themes.
    - Uses a reasonable number of bins based on data size.
    - Keeps histogram bars transparent (alpha from THEME["hist_alpha"]).
    """
    data = np.asarray(data)
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Background
    if THEME["dark"]:
        ax.set_facecolor("#111827")          # dark but not pure black
        fig.patch.set_facecolor("#111827")
    else:
        ax.set_facecolor(THEME["ax_facecolor"])
        fig.patch.set_facecolor(THEME["ax_facecolor"])

    # ----- choose number of bins -----
    # sqrt rule with some clamping so it's never silly
    n_bins = int(np.sqrt(len(data)) * 1.5)
    n_bins = max(5, min(40, n_bins))

    # ----- HISTOGRAM -----
    # ---------- ALWAYS USE TRANSPARENT HISTOGRAMS ---------- #
    transparency = 0.55  # universal alpha

    # ---------- VIRIDIS GRADIENT WITH TRUE TRANSPARENCY ---------- #
    if THEME["use_viridis"]:
        counts, bins = np.histogram(data, bins="auto", density=True)
        bin_widths = np.diff(bins)

        # Normalize bar heights for gradient mapping
        if len(counts) > 0 and np.max(counts) > np.min(counts):
            norm_counts = (counts - counts.min()) / (counts.max() - counts.min())
        else:
            norm_counts = np.zeros_like(counts)

        cmap = cm.get_cmap("viridis")

        # α = super transparent so viridis actually looks transparent
        viridis_alpha = 0.35

        for c, x_left, w, val in zip(counts, bins[:-1], bin_widths, norm_counts):
            bar_color = cmap(val)

            # Make the viridis color itself transparent
            bar_color = (bar_color[0], bar_color[1], bar_color[2], viridis_alpha)

            ax.bar(
                x_left + w / 2,
                c,
                width=w,
                color=bar_color,
                edgecolor=THEME["hist_edgecolor"],
                align="center",
                linewidth=0.7,
            )


    # ALL OTHER THEMES (Light, Dark, Neon, Seaborn)
    else:
        ax.hist(
            data,
            bins="auto",
            density=True,
            alpha=transparency,   # ← ALWAYS transparent
            color=THEME["hist_color"],
            edgecolor=THEME["hist_edgecolor"],
            linewidth=1.1,
        )


    # ----- PDF CURVE -----
    x = np.linspace(data.min(), data.max(), 400)
    pdf = dist.pdf(x, *params)

    if THEME.get("neon", False):
        # Neon glow: fat transparent line behind, bright line on top
        ax.plot(
            x,
            pdf,
            color=THEME["line_color_shadow"],
            linewidth=5.0,
            alpha=0.25,
        )
        ax.plot(
            x,
            pdf,
            color=THEME["line_color_main"],
            linewidth=2.6,
            label="Fitted PDF",
        )
    else:
        # Normal themed line with soft shadow
        ax.plot(
            x,
            pdf,
            color=THEME["line_color_shadow"],
            linewidth=3.2,
            alpha=0.30,
        )
        ax.plot(
            x,
            pdf,
            color=THEME["line_color_main"],
            linewidth=2.0,
            label="Fitted PDF",
        )

    # ----- LABELS & STYLING -----
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(title)

    ax.tick_params(axis="x", colors=THEME["text_color"])
    ax.tick_params(axis="y", colors=THEME["text_color"])
    ax.xaxis.label.set_color(THEME["text_color"])
    ax.yaxis.label.set_color(THEME["text_color"])
    ax.title.set_color(THEME["text_color"])

    spine_color = "#e5e7eb" if THEME["dark"] else "#333333"
    for spine in ax.spines.values():
        spine.set_color(spine_color)

    ax.legend(frameon=False, facecolor="none", edgecolor="none")
    fig.tight_layout()
    return fig



# ---------- SIDEBAR: DATA & OPTIONS ---------- #

st.sidebar.subheader("1. Data source")
input_mode = st.sidebar.radio(
    "How do you want to provide data?",
    ("Type numbers", "Upload CSV"),
    index=0
)

data = None

if input_mode == "Type numbers":
    text = st.sidebar.text_area(
        "Enter numbers (commas, spaces, or newlines):",
        "5, 6, 7, 8, 5, 4, 9, 3"
    )
    data = parse_manual_data(text)
else:
    uploaded = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)

        st.sidebar.write("CSV preview:")
        st.sidebar.dataframe(df.head())

        # Extract *all* numeric values in the entire sheet
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            st.sidebar.warning("No numeric columns found in this file.")
        else:
            mode = st.sidebar.radio(
                "Use data from:",
                ("All numeric data in the sheet", "Choose a single column"),
                index=0
            )

            if mode == "All numeric data in the sheet":
                # Flatten everything
                arr = numeric_df.to_numpy().ravel()
                arr = arr[~np.isnan(arr)]
                data = arr
            else:
                # Let user pick a column
                col = st.sidebar.selectbox(
                    "Select column:",
                    numeric_df.columns.tolist()
                )
                series = numeric_df[col]
                series = pd.to_numeric(series, errors="coerce")
                data = series.dropna().values

            st.sidebar.success(f"Loaded **{len(data)}** numeric data points.")



st.sidebar.subheader("2. Distribution")
dist_name = st.sidebar.selectbox(
    "Pick a model to fit:",
    list(DISTRIBUTIONS.keys())
)
dist = DISTRIBUTIONS[dist_name]

st.sidebar.subheader("3. Display")
show_raw_data = st.sidebar.checkbox("Show raw data table")
show_basic_stats = st.sidebar.checkbox("Show basic statistics", value=True)

# ---------- MAIN CONTENT ---------- #

st.title("Data Distribution Fitter")
st.caption(
    "Fit common probability distributions to your data, compare automatic and manual fits, "
    "and see how *center* and *spread* change the shape. Try different visual themes, "
    "including a neon cyber look."
)

if data is None or len(data) == 0:
    st.info("Add some data in the sidebar to get started.")
    st.stop()

# add icons to tab titles
tabs = st.tabs(["📈 Fit overview", "🎛️ Manual tuning", "📄 Data & summary"])

# ---------- TAB 1: FIT OVERVIEW ---------- #
with tabs[0]:
    st.header("Automatic fit")

    raw_params = dist.fit(data)
    shape_params, center, spread = split_params(raw_params)

    param_labels = [f"Shape {i+1}" for i in range(len(shape_params))]
    param_labels += ["Center", "Spread (width)"]
    param_values = list(shape_params) + [center, spread]

    col_metrics, col_plot = st.columns([1, 2])

    with col_metrics:
        st.subheader("Fitted parameters")
        param_df = pd.DataFrame({"Parameter": param_labels, "Value": param_values})
        st.markdown('<div class="param-table">', unsafe_allow_html=True)
        st.table(param_df)
        st.markdown("</div>", unsafe_allow_html=True)

        mse, mae = compute_fit_error(data, dist, raw_params)
        st.subheader("Fit quality")
        st.markdown('<div class="metric-box">', unsafe_allow_html=True)
        st.write(f"**Mean squared error (MSE):** {mse:.4e}")
        st.write(f"**Mean absolute error (MAE):** {mae:.4e}")
        st.markdown("</div>", unsafe_allow_html=True)

        if show_basic_stats:
            st.subheader("Data snapshot")
            st.markdown('<div class="metric-box">', unsafe_allow_html=True)
            st.write(f"Count: `{len(data)}`")
            st.write(f"Mean: `{np.mean(data):.4f}`")
            st.write(f"Std dev: `{np.std(data, ddof=1):.4f}`")
            st.markdown("</div>", unsafe_allow_html=True)

    with col_plot:
        fig_auto = make_histogram_with_pdf(
            data, dist, raw_params, f"Automatic fit: {dist_name}"
        )
        st.pyplot(fig_auto, use_container_width=True)

# ---------- TAB 2: MANUAL TUNING ---------- #
with tabs[1]:
    st.header("Manual tuning")

    st.write(
        "Use the sliders to **manually adjust** the distribution. "
        "**Center** shifts the curve left/right; **Spread** controls how wide or narrow it is. "
        "Shape parameters (if present) adjust skew and peak."
    )

    col_sliders, col_manual_plot = st.columns([1, 2])

    # reuse automatic fit params from tab 1
    raw_params = dist.fit(data)
    shape_params, center, spread = split_params(raw_params)

    with col_sliders:
        st.subheader("Adjust parameters")

        manual_shape_params = []
        for i, sp in enumerate(shape_params):
            span = max(abs(sp), 1.0)
            min_val = sp - span
            max_val = sp + span
            manual_sp = st.slider(
                f"Shape {i+1}",
                float(min_val),
                float(max_val),
                float(sp),
                step=float(span / 50.0)
            )
            manual_shape_params.append(manual_sp)

        center_span = max(abs(center), 1.0)
        center_min = center - center_span
        center_max = center + center_span
        manual_center = st.slider(
            "Center (horizontal shift)",
            float(center_min),
            float(center_max),
            float(center),
            step=float(center_span / 50.0)
        )

        spread_span = max(abs(spread), 1.0)
        spread_min = max(1e-6, spread - spread_span)
        spread_max = spread + spread_span
        manual_spread = st.slider(
            "Spread (width)",
            float(spread_min),
            float(spread_max),
            float(spread),
            step=float(spread_span / 50.0)
        )

        manual_params = tuple(manual_shape_params + [manual_center, manual_spread])

        auto_mse, auto_mae = compute_fit_error(data, dist, raw_params)
        manual_mse, manual_mae = compute_fit_error(data, dist, manual_params)

        st.subheader("Fit comparison")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Automatic MSE", f"{auto_mse:.3e}")
            st.metric("Automatic MAE", f"{auto_mae:.3e}")
        with c2:
            st.metric(
                "Manual MSE",
                f"{manual_mse:.3e}",
                delta=f"{manual_mse - auto_mse:.1e} vs auto"
            )
            st.metric(
                "Manual MAE",
                f"{manual_mae:.3e}",
                delta=f"{manual_mae - auto_mae:.1e} vs auto"
            )

    with col_manual_plot:
        fig_manual = make_histogram_with_pdf(
            data, dist, manual_params, f"Manual fit: {dist_name}"
        )
        st.pyplot(fig_manual, use_container_width=True)

# ---------- TAB 3: DATA & SUMMARY ---------- #
with tabs[2]:
    st.header("Data & summary")

    if show_raw_data:
        st.subheader("Raw data")
        st.dataframe(pd.DataFrame({"values": data}))

    st.subheader("Descriptive statistics")
    stats_df = pd.DataFrame(
        {
            "Statistic": ["Count", "Mean", "Std dev", "Min", "Max"],
            "Value": [
                len(data),
                np.mean(data),
                np.std(data, ddof=1),
                np.min(data),
                np.max(data),
            ],
        }
    )
    st.table(stats_df)

    st.subheader("What this app is doing")
    st.markdown(
        """
        - You provide a **sample of data** (typed or from a CSV).
        - For the chosen distribution, SciPy finds the best-fitting parameters.
        - Those parameters include:
            - One or more **Shape** parameters (distribution-specific),
            - A **Center** (previously *loc*, shifts the curve left/right),
            - A **Spread** (previously *scale*, controls width).
        - The histogram is drawn as a density, and the model PDF is plotted on top.
        - Error metrics (MSE and MAE) measure the difference between data and model.
        - In **Manual tuning**, you can experiment with the parameters yourself and see how the curve moves.
        """
    )
