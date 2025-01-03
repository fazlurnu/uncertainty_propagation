import os
import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.mixture import GaussianMixture
from autonomous_separation import *

from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
#  Constants and Parameters
# --------------------------------------------------------------------------
X_OWN = 0
Y_OWN = 0
TRK_OWN = 0             # Heading of ownship in degrees
GS_OWN = 20             # Groundspeed of ownship

RPZ = 50                # Radius Protected Zone (example)
TLOSH = 15              # Time to Loss of Separation
SIGMA = 15              # Std. deviation for positional noise
NB_SAMPLES = 10_000     # Number of Monte Carlo samples
GMM_SEED = 42           # Random state for GMM initialization

DCPA_START = 0
DCPA_END = 50
DCPA_STEP = 5

DPSI_START = 0
DPSI_END = 2
DPSI_STEP = 2

SPD_INTRUDER = 15       # Intruder speed

GS_SIGMA = 0            # Not used in current code, but you can adapt if needed
HDG_SIGMA = 0           # Not used in current code, but you can adapt if needed

OUTPUT_DIR = "dataframes_pos_uncertainty"

# Grid for DBSCAN hyperparameters
EPS_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]
MIN_SAMPLES_GRID = [3, 5, 7, 10]

def find_best_dbscan_params(data, eps_values, min_samples_values):
    """
    Search over a grid of eps and min_samples for DBSCAN,
    choosing the combination that yields the best silhouette score.
    
    Parameters
    ----------
    data : np.ndarray
        2D array of shape (n_samples, n_features).
    eps_values : list or iterable
        Candidate values for DBSCAN 'eps'.
    min_samples_values : list or iterable
        Candidate values for DBSCAN 'min_samples'.

    Returns
    -------
    best_params : tuple
        (best_eps, best_min_samples)
    best_score : float
        The highest silhouette score found.
    """
    best_score = -1
    best_params = (None, None)

    for eps in eps_values:
        for min_samp in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samp)
            labels = dbscan.fit_predict(data)

            # If there's only one cluster (or all outliers),
            # silhouette is not defined => skip
            unique_labels = set(labels)
            if len(unique_labels) < 2:
                continue

            score = silhouette_score(data, labels)
            if score > best_score:
                best_score = score
                best_params = (eps, min_samp)

    return best_params, best_score

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # ------------------------------------------
    # Prepare a list to store the simulation logs
    # ------------------------------------------
    results = []

    for dcpa_val in range(DCPA_START, DCPA_END, DCPA_STEP):
        for dpsi_val in range(DPSI_START, DPSI_END, DPSI_STEP):
            
            # 1) Create a conflict scenario
            x_int, y_int, trk_int, gs_int = cre_conflict(
                X_OWN, Y_OWN, TRK_OWN, GS_OWN, 
                dpsi_val, dcpa_val, TLOSH, SPD_INTRUDER, RPZ
            )

            # 2) Generate noisy samples for positions
            x_o_noise, y_o_noise = create_pos_noise_samples(
                X_OWN, Y_OWN, nb_samples=NB_SAMPLES, sigma=SIGMA
            )
            x_i_noise, y_i_noise = create_pos_noise_samples(
                x_int, y_int, nb_samples=NB_SAMPLES, sigma=SIGMA
            )

            # 3) Generate noisy samples for heading & speed
            hdg_ownship = np.random.normal(TRK_OWN, HDG_SIGMA, NB_SAMPLES)
            gs_ownship  = np.random.normal(GS_OWN, GS_SIGMA, NB_SAMPLES)
            hdg_intruder = np.random.normal(trk_int, HDG_SIGMA, NB_SAMPLES)
            gs_intruder  = np.random.normal(gs_int, GS_SIGMA, NB_SAMPLES)

            # 4) Build DataFrame
            df = pd.DataFrame({
                'x_own_true':  X_OWN,
                'y_own_true':  Y_OWN,
                'x_own_noise': x_o_noise,
                'y_own_noise': y_o_noise,
                'hdg_own':     hdg_ownship,
                'gs_own':      gs_ownship,

                'x_int_true':  x_int,
                'y_int_true':  y_int,
                'x_int_noise': x_i_noise,
                'y_int_noise': y_i_noise,
                'hdg_int':     hdg_intruder,
                'gs_int':      gs_intruder
            })

            # 5) Shapely Points
            df['pos_ownship'] = [
                Point(x, y) for x, y in zip(df['x_own_noise'], df['y_own_noise'])
            ]
            df['pos_intruder'] = [
                Point(x, y) for x, y in zip(df['x_int_noise'], df['y_int_noise'])
            ]

            # 6) Conflict detection
            df[['dx', 'dy', 'tin', 'tout', 'dcpa', 'is_conflict']] = df.apply(
                lambda row: pd.Series(detect_conflict(row)), axis=1
            )

            # 7) Compute resolution velocities
            df[['vx_vo', 'vy_vo']] = df.apply(
                lambda row: pd.Series(conf_reso_VO(row)), axis=1
            )
            df[['vx_mvp', 'vy_mvp']] = df.apply(
                lambda row: pd.Series(conf_reso_MVP(row)), axis=1
            )

            # 8) Keep only rows with conflict
            df_conflicts = df.loc[df['is_conflict']].copy()

            # 9) Save CSV (optional)
            out_filename = f"pos_uncertainty_{dcpa_val}_{dpsi_val}.csv"
            csv_path = os.path.join(OUTPUT_DIR, out_filename)
            df_conflicts.to_csv(csv_path, index=False)

            # -------------------------------------------------------------
            #  We'll store results for VO and MVP in local variables, then
            #  append them to the "results" list as one row/dict.
            # -------------------------------------------------------------
            nb_cluster_vo = 0
            weights_vo = []
            means_vo   = []

            nb_cluster_mvp = 0
            weights_mvp = []
            means_mvp   = []

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #  DBSCAN => GMM for VO data
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            X_vo = df_conflicts[['vx_vo', 'vy_vo']].to_numpy()
            if len(X_vo) > 1:
                scaler_vo = StandardScaler()
                X_vo_scaled = scaler_vo.fit_transform(X_vo)

                # A) Find best DBSCAN hyperparameters
                (best_eps_vo, best_min_vo), best_score_vo = find_best_dbscan_params(
                    X_vo_scaled,
                    eps_values=EPS_GRID,
                    min_samples_values=MIN_SAMPLES_GRID
                )

                # Decide on n_components from DBSCAN
                if best_eps_vo is None:
                    # No valid silhouette => only 1 cluster formed (or 0)
                    n_components_vo = 1
                else:
                    dbscan_vo = DBSCAN(eps=best_eps_vo, min_samples=best_min_vo)
                    vo_labels = dbscan_vo.fit_predict(X_vo_scaled)
                    unique_vo_labels = set(vo_labels)
                    if -1 in unique_vo_labels:
                        unique_vo_labels.remove(-1)
                    n_components_vo = len(unique_vo_labels)
                    if n_components_vo == 0:
                        n_components_vo = 1

                # B) Fit GMM
                gmm_vo = GaussianMixture(n_components=n_components_vo, random_state=42)
                gmm_vo.fit(X_vo_scaled)

                # Extract GMM parameters
                nb_cluster_vo = n_components_vo
                weights_vo = gmm_vo.weights_.tolist()
                means_vo   = gmm_vo.means_.tolist()

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            #  DBSCAN => GMM for MVP data
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            X_mvp = df_conflicts[['vx_mvp', 'vy_mvp']].to_numpy()
            if len(X_mvp) > 1:
                scaler_mvp = StandardScaler()
                X_mvp_scaled = scaler_mvp.fit_transform(X_mvp)

                # A) Find best DBSCAN hyperparameters
                (best_eps_mvp, best_min_mvp), best_score_mvp = find_best_dbscan_params(
                    X_mvp_scaled,
                    eps_values=EPS_GRID,
                    min_samples_values=MIN_SAMPLES_GRID
                )

                # Decide on n_components from DBSCAN
                if best_eps_mvp is None:
                    n_components_mvp = 1
                else:
                    dbscan_mvp = DBSCAN(eps=best_eps_mvp, min_samples=best_min_mvp)
                    mvp_labels = dbscan_mvp.fit_predict(X_mvp_scaled)
                    unique_mvp_labels = set(mvp_labels)
                    if -1 in unique_mvp_labels:
                        unique_mvp_labels.remove(-1)
                    n_components_mvp = len(unique_mvp_labels)
                    if n_components_mvp == 0:
                        n_components_mvp = 1

                # B) Fit GMM
                gmm_mvp = GaussianMixture(n_components=n_components_mvp, random_state=42)
                gmm_mvp.fit(X_mvp_scaled)

                # Extract GMM parameters
                nb_cluster_mvp = n_components_mvp
                weights_mvp = gmm_mvp.weights_.tolist()
                means_mvp   = gmm_mvp.means_.tolist()

            # -------------------------------------------------------------
            #  Append the summary for this (dcpa_val, dpsi_val) iteration
            # -------------------------------------------------------------
            results.append({
                "dcpa_val": dcpa_val,
                "dpsi_val": dpsi_val,
                "nb_cluster_vo": nb_cluster_vo,
                "cluster_weights_vo": weights_vo,   # list of floats
                "cluster_means_vo": means_vo,       # list of lists
                "nb_cluster_mvp": nb_cluster_mvp,
                "cluster_weights_mvp": weights_mvp, # list of floats
                "cluster_means_mvp": means_mvp      # list of lists
            })

    # ------------------------------------------------------------
    #  After ALL loops, create a DataFrame of our results
    # ------------------------------------------------------------
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(OUTPUT_DIR, "dataframe_results.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"\Dataframe logging saved to: {results_csv_path}")

if __name__ == "__main__":
    main()