import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astroquery.gaia import Gaia
from sklearn.mixture import GaussianMixture

def query_gaia_archive(limit=15000):
    """
    Executes an ADQL query via TAP against the ESA Gaia DR3 archive
    to retrieve high-precision astrometric and photometric telemetry.
    """
    print(f"1. Querying ESA Gaia DR3 archive (fetching {limit} high-parallax stars)...")
    
    adql_query = f"""
    SELECT TOP {limit}
        source_id, ra, dec, parallax, parallax_error,
        phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
        bp_rp
    FROM gaiadr3.gaia_source
    WHERE parallax > 10.0
      AND parallax_over_error > 10.0
      AND phot_g_mean_flux_over_error > 50.0
      AND bp_rp IS NOT NULL
    """
    
    job = Gaia.launch_job_async(adql_query)
    results = job.get_results()
    df = results.to_pandas()
    print(f"✓ Retrieved {len(df)} stellar sources with high astrometric fidelity.")
    return df

def derive_astrophysical_parameters(df):
    """
    Derives physical distance (parsecs) and absolute magnitude (M_G)
    using the distance modulus formula.
    """
    print("2. Deriving distance moduli and absolute bolometric magnitudes...")
    
    # Distance in parsecs: d = 1000 / parallax (mas)
    df["distance_pc"] = 1000.0 / df["parallax"]
    
    # Absolute magnitude: M_G = G + 5 + 5 * log10(parallax / 1000)
    df["abs_g_mag"] = df["phot_g_mean_mag"] + 5.0 + 5.0 * np.log10(df["parallax"] / 1000.0)
    
    # Clean non-finite entries and apply realistic local stellar bounds
    df_clean = df.dropna(subset=["bp_rp", "abs_g_mag"]).copy()
    df_clean = df_clean[
        (df_clean["bp_rp"] > -0.5) & (df_clean["bp_rp"] < 4.0) &
        (df_clean["abs_g_mag"] > -5.0) & (df_clean["abs_g_mag"] < 18.0)
    ]
    
    print(f"✓ Processed {len(df_clean)} verified stars ready for clustering.")
    return df_clean

def classify_stellar_populations(df):
    """
    Applies an unsupervised Gaussian Mixture Model (GMM) to segment 
    stars into Main Sequence, Red Giants, and White Dwarfs.
    """
    print("3. Executing Gaussian Mixture Model (k=3) for branch clustering...")
    X = df[["bp_rp", "abs_g_mag"]].values
    
    gmm = GaussianMixture(n_components=3, random_state=42)
    df["cluster"] = gmm.fit_predict(X)
    
    cluster_means = df.groupby("cluster")[["bp_rp", "abs_g_mag"]].mean()
    
    # White Dwarfs: high magnitude (dim), low color index (hot/blue)
    wd_cluster = cluster_means.sort_values(by="abs_g_mag", ascending=False).index[0]
    
    # Red Giants: low magnitude (luminous), high color index (cool/red)
    rem = cluster_means.drop(index=wd_cluster)
    rg_cluster = rem.sort_values(by="bp_rp", ascending=False).index[0]
    
    # Remaining cluster: Main Sequence
    ms_cluster = rem.drop(index=rg_cluster).index[0]
    
    label_map = {
        ms_cluster: "Main Sequence",
        rg_cluster: "Red Giant Branch",
        wd_cluster: "White Dwarfs"
    }
    df["stellar_class"] = df["cluster"].map(label_map)
    print("✓ Stellar populations labeled: Main Sequence, Red Giants, White Dwarfs.")
    return df

def plot_hertzsprung_russell(df):
    """
    Generates publication-quality Hertzsprung-Russell Diagram 
    with inverted astronomical magnitude axis and spectral styling.
    """
    print("4. Rendering Hertzsprung-Russell Diagram...")
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0a0e14")
    ax.set_facecolor("#0a0e14")
    
    colors = {
        "Main Sequence": "#ffcc00",
        "Red Giant Branch": "#ff4d4d",
        "White Dwarfs": "#00d4ff"
    }
    
    for label, group in df.groupby("stellar_class"):
        ax.scatter(
            group["bp_rp"], 
            group["abs_g_mag"], 
            s=2.5, 
            c=colors[label], 
            alpha=0.6, 
            edgecolors="none", 
            label=f"{label} ({len(group):,} stars)"
        )
    
    # Astronomical standard: Invert y-axis (brighter stars have lower/negative magnitudes)
    ax.invert_yaxis()
    
    ax.set_title("ESA Gaia DR3 Hertzsprung-Russell Stellar Evolution Engine", color="white", fontsize=13, pad=15, fontweight="bold")
    ax.set_xlabel(r"Color Index: $G_{BP} - G_{RP}$ (Blue/Hot $\leftarrow$ Temperature $\rightarrow$ Red/Cool)", color="#d0d0d0", fontsize=11)
    ax.set_ylabel(r"Absolute Magnitude: $M_G$ (Luminosity $\rightarrow$ Brighter)", color="#d0d0d0", fontsize=11)
    
    ax.tick_params(colors="white", which="both")
    for spine in ax.spines.values():
        spine.set_color("#444444")
        
    ax.grid(True, linestyle=":", alpha=0.3, color="#888888")
    
    legend = ax.legend(loc="upper right", facecolor="#141a22", edgecolor="#444444")
    for text in legend.get_texts():
        text.set_color("white")
        
    plt.tight_layout()
    plt.savefig("gaia_hr_diagram.png", dpi=300, facecolor=fig.get_facecolor())
    print("✓ Diagram saved as 'gaia_hr_diagram.png'.")
    plt.show()

if __name__ == "__main__":
    df_raw = query_gaia_archive(limit=15000)
    df_phys = derive_astrophysical_parameters(df_raw)
    df_classified = classify_stellar_populations(df_phys)
    plot_hertzsprung_russell(df_classified)