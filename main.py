import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from shapely.geometry import Point, LineString, Polygon
from shapely.affinity import translate

# Scikit-learn imports
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

from time import gmtime, strftime

# Replace these with your actual import statements if different
from autonomous_separation import (
    cre_conflict,
    create_pos_noise_samples,
    detect_conflict,
    conf_reso_VO,
    conf_reso_MVP,
    VO,
    MVP,
    get_cc_tp
)


class UncertaintyCaseSelector:
    """
    Handles user input for determining which uncertainties (position, heading, speed)
    and which source (ownship, intruder, both) are active.
    """
    def __init__(self):
        self.case_title_idx = None
        self.source_of_uncertainty_idx = None

    @staticmethod
    def binary_to_case_title(binary_input: int) -> str:
        """
        Convert a decimal integer (1-7) corresponding to a three-bit binary string
        into a descriptive string for type of uncertainty (pos, hdg, speed).
        """
        cases = {
            1: "position only",        # 001
            2: "heading only",         # 010
            4: "speed only",           # 100
            3: "position and heading", # 011
            5: "position and speed",   # 101
            6: "heading and speed",    # 110
            7: "all"                   # 111
        }
        return cases.get(binary_input, "Invalid case")

    @staticmethod
    def binary_to_uncertainty_source(binary_input: int) -> str:
        """
        Convert a decimal integer (1-3) corresponding to a two-bit binary string
        into a descriptive string for source of uncertainty (ownship, intruder, both).
        """
        sources = {
            1: "ownship only",   # 01
            2: "intruder only",  # 10
            3: "both"            # 11
        }
        return sources.get(binary_input, "Invalid case")

    def select_case_title(self):
        """
        Prompt user to enter a three-bit binary string to select the uncertainty type
        (position, heading, speed) and store both the descriptive string and index.
        """
        print("Select case title by activating the binary")
        print("  001 for position only")
        print("  010 for heading only")
        print("  100 for speed only")
        print("  011 for position and heading")
        print("  101 for position and speed")
        print("  110 for heading and speed")
        print("  111 for all")

        binary_input = input("Case title here (three-digit binary): ").strip()
        if len(binary_input) != 3 or not all(bit in "01" for bit in binary_input):
            print("Invalid input. Please enter a three-digit binary number.")
            return None, None

        self.case_title_idx = int(binary_input, 2)
        return (self.binary_to_case_title(self.case_title_idx), self.case_title_idx)

    def select_source_of_uncertainty(self):
        """
        Prompt user to enter a two-bit binary string to select the source of uncertainty
        (ownship, intruder, both) and store both the descriptive string and index.
        """
        print("Select the mixture of source of uncertainty by activating the binary")
        print("  01 for ownship only")
        print("  10 for intruder only")
        print("  11 for both")

        binary_input = input("Source of uncertainty here (two-digit binary): ").strip()
        if len(binary_input) != 2 or not all(bit in "01" for bit in binary_input):
            print("Invalid input. Please enter a two-digit binary number.")
            return None, None

        self.source_of_uncertainty_idx = int(binary_input, 2)
        return (
            self.binary_to_uncertainty_source(self.source_of_uncertainty_idx),
            self.source_of_uncertainty_idx
        )


class ConflictClustering:
    """
    Handles the clustering logic (DBSCAN => GMM) for conflict data, along with
    saving CSVs and storing results. Also includes constants for:
      - OUTPUT_DIR: where data & plots get saved
      - DBSCAN hyperparameter grids
    """
    OUTPUT_DIR = "dataframes_pos_uncertainty"

    # Hyperparameter grids for DBSCAN
    EPS_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]
    MIN_SAMPLES_GRID = [3, 5, 7, 10]

    @staticmethod
    def find_best_dbscan_params(data, eps_values, min_samples_values):
        """
        Search over a grid of eps and min_samples for DBSCAN,
        choosing the combination that yields the best silhouette score.
        If there's only one cluster (or all outliers), silhouette is not defined => skip.
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

    def analyze_conflicts(
        self, 
        df_conflicts: pd.DataFrame,
        dcpa_val: int,
        dpsi_val: int,
        case_idx: str,
        source_of_uncertainty_idx: str,
        spd: float
    ):
        """
        1) Saves the conflict-data CSV to OUTPUT_DIR using the naming pattern:
           {case_title}_{source_of_uncertainty}_{spd}_{dpsi_val}_{dcpa_val}.csv
        2) Performs DBSCAN => GMM for both VO and MVP velocities.
        3) Returns a dict summarizing clustering results (e.g., # clusters, weights, means).
        """
        # Ensure the output directory exists
        if not os.path.isdir(self.OUTPUT_DIR):
            os.makedirs(self.OUTPUT_DIR, exist_ok=True)

        # -----------------------------------------------------------
        # CHANGES: Updated CSV filename to include the required fields
        # -----------------------------------------------------------
        out_filename = f"{case_idx}_{source_of_uncertainty_idx}_{spd}_{dpsi_val}_{dcpa_val}.csv"
        csv_path = os.path.join(self.OUTPUT_DIR, out_filename)
        df_conflicts.to_csv(csv_path, index=False)

        # Initialize cluster results
        nb_cluster_vo, weights_vo, means_vo = 0, [], []
        nb_cluster_mvp, weights_mvp, means_mvp = 0, [], []

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #  DBSCAN => GMM for VO data
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        X_vo = df_conflicts[['vx_vo', 'vy_vo']].to_numpy()
        if len(X_vo) > 1:
            scaler_vo = StandardScaler()
            X_vo_scaled = scaler_vo.fit_transform(X_vo)

            (best_eps_vo, best_min_vo), best_score_vo = self.find_best_dbscan_params(
                X_vo_scaled,
                eps_values=self.EPS_GRID,
                min_samples_values=self.MIN_SAMPLES_GRID
            )

            # Decide on n_components for GMM from DBSCAN
            if best_eps_vo is None:
                # No valid silhouette => only 1 cluster formed (or 0)
                n_components_vo = 1
            else:
                dbscan_vo = DBSCAN(eps=best_eps_vo, min_samples=best_min_vo)
                vo_labels = dbscan_vo.fit_predict(X_vo_scaled)
                unique_vo_labels = set(vo_labels)
                if -1 in unique_vo_labels:
                    unique_vo_labels.remove(-1)
                n_components_vo = len(unique_vo_labels) if len(unique_vo_labels) > 0 else 1

            # Fit GMM
            gmm_vo = GaussianMixture(n_components=n_components_vo, random_state=42)
            gmm_vo.fit(X_vo_scaled)

            # Extract GMM parameters
            nb_cluster_vo = n_components_vo
            weights_vo = gmm_vo.weights_.tolist()
            means_vo = gmm_vo.means_.tolist()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        #  DBSCAN => GMM for MVP data
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        X_mvp = df_conflicts[['vx_mvp', 'vy_mvp']].to_numpy()
        if len(X_mvp) > 1:
            scaler_mvp = StandardScaler()
            X_mvp_scaled = scaler_mvp.fit_transform(X_mvp)

            (best_eps_mvp, best_min_mvp), best_score_mvp = self.find_best_dbscan_params(
                X_mvp_scaled,
                eps_values=self.EPS_GRID,
                min_samples_values=self.MIN_SAMPLES_GRID
            )

            # Decide on n_components for GMM from DBSCAN
            if best_eps_mvp is None:
                n_components_mvp = 1
            else:
                dbscan_mvp = DBSCAN(eps=best_eps_mvp, min_samples=best_min_mvp)
                mvp_labels = dbscan_mvp.fit_predict(X_mvp_scaled)
                unique_mvp_labels = set(mvp_labels)
                if -1 in unique_mvp_labels:
                    unique_mvp_labels.remove(-1)
                n_components_mvp = len(unique_mvp_labels) if len(unique_mvp_labels) > 0 else 1

            # Fit GMM
            gmm_mvp = GaussianMixture(n_components=n_components_mvp, random_state=42)
            gmm_mvp.fit(X_mvp_scaled)

            # Extract GMM parameters
            nb_cluster_mvp = n_components_mvp
            weights_mvp = gmm_mvp.weights_.tolist()
            means_mvp = gmm_mvp.means_.tolist()

        # Return all info as a dictionary
        return {
            "dcpa_val": dcpa_val,
            "dpsi_val": dpsi_val,
            "nb_cluster_vo": nb_cluster_vo,
            "cluster_weights_vo": weights_vo,
            "cluster_means_vo": means_vo,
            "nb_cluster_mvp": nb_cluster_mvp,
            "cluster_weights_mvp": weights_mvp,
            "cluster_means_mvp": means_mvp
        }


class ConflictResolutionSimulation:
    """
    Encapsulates all simulation parameters and methods for running
    conflict-resolution scenarios under various uncertainties.
    """

    def __init__(self):
        # -- DEFAULTS / INITIAL SETUP --
        self.x_own = 0
        self.y_own = 0
        self.hdg_own = 0
        self.gs_own = 20

        # Loop parameters
        self.gs_int = 15 ## this is the speed of the intruder
        self.tlosh = 15
        self.rpz = 50
        self.dcpa_start = 0
        self.dcpa_end = 49
        self.dcpa_delta = 5  # With start=0, end=4, delta=5 => only dcpa=0
        self.dpsi_start = 0
        self.dpsi_end = 181
        self.dpsi_delta = 10

        # For final plot axis limits
        self.vy_init = self.gs_own * np.sin(np.radians(self.hdg_own))
        self.vx_init = self.gs_own * np.cos(np.radians(self.hdg_own))

        self.nb_samples = 5000
        self.alpha_uncertainty = 0.4

        # Uncertainty switches (defaults to False)
        self.pos_uncertainty_on = False
        self.hdg_uncertainty_on = False
        self.spd_uncertainty_on = False
        self.src_ownship_on = False
        self.src_intruder_on = False

        # Noise parameters
        self.sigma = 0
        self.hdg_sigma_ownship = 0
        self.hdg_sigma_intruder = 0
        self.gs_sigma_ownship = 0
        self.gs_sigma_intruder = 0

        # For user selections
        self.case_title_selected = None
        self.case_title_idx = None
        self.source_of_uncertainty = None
        self.source_of_uncertainty_idx = None

        # Clustering logic & results
        self.clustering = ConflictClustering()
        self.results = []

    def set_uncertainty_switches(self, case_title_idx: int, source_of_uncertainty_idx: int):
        """
        Determine which uncertainties (position, heading, speed) apply,
        and whether they apply to ownship, intruder, or both.
        """
        # 3-bit input for position / heading / speed
        self.pos_uncertainty_on = bool(case_title_idx & 0b001)  # position bit
        self.hdg_uncertainty_on = bool(case_title_idx & 0b010)  # heading bit
        self.spd_uncertainty_on = bool(case_title_idx & 0b100)  # speed bit

        # 2-bit input for ownship / intruder
        self.src_ownship_on = bool(source_of_uncertainty_idx & 0b01)
        self.src_intruder_on = bool(source_of_uncertainty_idx & 0b10)

    def set_noise_parameters(self):
        """
        Adjust noise parameters (standard deviations) based on which
        uncertainties and sources are active.
        """
        # Position noise
        if self.pos_uncertainty_on:
            self.sigma = 15  # Example standard deviation for position

        # Heading noise
        if self.hdg_uncertainty_on:
            if self.src_ownship_on:
                self.hdg_sigma_ownship = 5
            if self.src_intruder_on:
                self.hdg_sigma_intruder = 5

        # Speed noise
        if self.spd_uncertainty_on:
            if self.src_ownship_on:
                self.gs_sigma_ownship = 5
            if self.src_intruder_on:
                self.gs_sigma_intruder = 5

    def run_simulation(self):
        """
        Main simulation loop. Creates conflicts, generates noisy samples,
        detects conflicts, clusters them (DBSCAN => GMM), and visualizes resolutions.
        """
        for dcpa_val in range(self.dcpa_start, self.dcpa_end, self.dcpa_delta):
            for dpsi_val in range(self.dpsi_start, self.dpsi_end, self.dpsi_delta):
                # --- 1. Intruder scenario creation ---
                x_int, y_int, hdg_int, gs_int = cre_conflict(
                    self.x_own, self.y_own, self.hdg_own, self.gs_own,
                    dpsi_val, dcpa_val, self.tlosh, self.gs_int, self.rpz
                )

                # --- 2. Create noisy positions ---
                x_o, y_o = create_pos_noise_samples(
                    self.x_own, self.y_own, self.nb_samples, sigma=self.sigma
                )
                x_i, y_i = create_pos_noise_samples(
                    x_int, y_int, self.nb_samples, sigma=self.sigma
                )

                # --- 3. Create noisy heading & speed for ownship/intruder ---
                hdg_ownship = np.random.normal(
                    self.hdg_own, self.hdg_sigma_ownship, self.nb_samples
                )
                gs_ownship = np.random.normal(
                    self.gs_own, self.gs_sigma_ownship, self.nb_samples
                )
                hdg_intruder = np.random.normal(
                    hdg_int, self.hdg_sigma_intruder, self.nb_samples
                )
                gs_intruder = np.random.normal(
                    gs_int, self.gs_sigma_intruder, self.nb_samples
                )

                # --- 4. Build DataFrame of samples ---
                df = pd.DataFrame({
                    'x_own_true': self.x_own,
                    'y_own_true': self.y_own,
                    'x_own_noise': x_o,
                    'y_own_noise': y_o,
                    'hdg_own': hdg_ownship,
                    'gs_own': gs_ownship,
                    'x_int_true': x_int,
                    'y_int_true': y_int,
                    'x_int_noise': x_i,
                    'y_int_noise': y_i,
                    'hdg_int': hdg_intruder,
                    'gs_int': gs_intruder
                })

                df['pos_ownship'] = [
                    Point(a, b) for a, b in zip(df['x_own_noise'], df['y_own_noise'])
                ]
                df['pos_intruder'] = [
                    Point(a, b) for a, b in zip(df['x_int_noise'], df['y_int_noise'])
                ]

                # --- 5. Detect conflicts, apply conflict resolution ---
                df[['dx', 'dy', 'tin', 'tout', 'dcpa', 'is_conflict']] = df.apply(
                    lambda row: pd.Series(detect_conflict(row)),
                    axis=1
                )
                df[['vx_vo', 'vy_vo']] = df.apply(
                    lambda row: pd.Series(conf_reso_VO(row)),
                    axis=1
                )
                df[['vx_mvp', 'vy_mvp']] = df.apply(
                    lambda row: pd.Series(conf_reso_MVP(row)),
                    axis=1
                )

                # --- 6. Filter to only rows in conflict for clustering & plotting ---
                df_conflict = df.loc[df['is_conflict']].copy()
                if df_conflict.empty:
                    # No conflicts => skip
                    continue

                # --- 7. Perform clustering & store results ---
                cluster_results = self.clustering.analyze_conflicts(
                    df_conflict,
                    dcpa_val,
                    dpsi_val,
                    self.case_title_idx,       # <-- pass along
                    self.source_of_uncertainty_idx,     # <-- pass along
                    self.gs_int                        # <-- pass along
                )
                self.results.append(cluster_results)

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                #  8. PLOTTING
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Prepare data for plotting (only rows in conflict).
                df_vo = df_conflict[['vx_vo', 'vy_vo']].copy()
                df_vo['vx'] = df_conflict['vx_vo']
                df_vo['vy'] = df_conflict['vy_vo']
                df_vo['Conf Reso'] = 'VO'

                df_mvp = df_conflict[['vx_mvp', 'vy_mvp']].copy()
                df_mvp['vx'] = df_conflict['vx_mvp']
                df_mvp['vy'] = df_conflict['vy_mvp']
                df_mvp['Conf Reso'] = 'MVP'

                df_plot = pd.concat([df_vo, df_mvp], ignore_index=True)

                # Create jointplot
                f = sns.jointplot(
                    data=df_plot,
                    x="vy", y="vx", hue="Conf Reso",
                    kind="scatter", alpha=self.alpha_uncertainty,
                    palette=['tab:blue', 'tab:orange']
                )
                f.set_axis_labels("Resolution $V_{y-body}$ [kts]", "Resolution $V_{x-body}$ [kts]")

                # Set x and y limits around the ownship velocity
                max_offset_vy = 10
                max_offset_vx = 10
                f.ax_joint.set_xlim(self.vy_init - max_offset_vy, self.vy_init + max_offset_vy)
                f.ax_joint.set_ylim(self.vx_init - max_offset_vx, self.vx_init + max_offset_vx)

                # --- Plot the VO region, if possible ---
                tp_1, tp_2 = get_cc_tp(Point(self.x_own, self.y_own),
                                       Point(x_int, y_int), self.rpz)
                int_vel = Point(gs_int * np.cos(np.radians(hdg_int)),
                                gs_int * np.sin(np.radians(hdg_int)))
                ownship_pos = Point(self.x_own, self.y_own)

                if tp_1 is not None and tp_2 is not None:
                    vo_0 = translate(ownship_pos, xoff=int_vel.x, yoff=int_vel.y)
                    vo_1 = translate(tp_1, xoff=int_vel.x, yoff=int_vel.y)
                    vo_2 = translate(tp_2, xoff=int_vel.x, yoff=int_vel.y)

                    dx_1, dy_1 = (vo_1.x - vo_0.x), (vo_1.y - vo_0.y)
                    dx_2, dy_2 = (vo_2.x - vo_0.x), (vo_2.y - vo_0.y)

                    # Extend lines for visualization
                    extension_length = 100
                    extended_endpoint_1 = Point(vo_1.x + dx_1 * extension_length,
                                                vo_1.y + dy_1 * extension_length)
                    extended_endpoint_2 = Point(vo_2.x + dx_2 * extension_length,
                                                vo_2.y + dy_2 * extension_length)

                    f.ax_joint.plot([vo_0.y, extended_endpoint_1.y],
                                    [vo_0.x, extended_endpoint_1.x],
                                    color='k')
                    f.ax_joint.plot([vo_0.y, extended_endpoint_2.y],
                                    [vo_0.x, extended_endpoint_2.x],
                                    color='k')

                    # Fill polygon between the two lines
                    vo_line_1 = LineString([vo_0, vo_1])
                    vo_line_2 = LineString([vo_0, vo_2])
                    coords_1 = list(vo_line_1.coords)
                    coords_2 = list(vo_line_2.coords)
                    polygon_coords = coords_1 + coords_2[::-1]
                    polygon = Polygon(polygon_coords)

                    ax = f.ax_joint
                    x_poly, y_poly = polygon.exterior.xy
                    ax.fill(y_poly, x_poly, alpha=0.3, fc='red', zorder=-1)

                # --- Plot true resolutions for VO and MVP ---
                vx_true_vo, vy_true_vo = VO(
                    Point(self.x_own, self.y_own), self.gs_own, self.hdg_own,
                    Point(x_int, y_int), gs_int, hdg_int,
                    self.rpz, method=0
                )
                _, vx_true_mvp, vy_true_mvp = MVP(
                    Point(self.x_own, self.y_own), self.gs_own, self.hdg_own,
                    Point(x_int, y_int), gs_int, hdg_int,
                    self.rpz
                )

                handles, labels = f.ax_joint.get_legend_handles_labels()
                f.ax_joint.legend(handles=handles, labels=labels, loc='upper right', bbox_to_anchor=(1.25, 1.2))

                # Mark lines/points for VO
                f.ax_joint.axhline(vx_true_vo, color='tab:blue', linestyle='--',
                                   linewidth=1.5, alpha=0.7)
                f.ax_joint.axvline(vy_true_vo, color='tab:blue', linestyle='--',
                                   linewidth=1.5, alpha=0.7)
                f.ax_joint.scatter(vy_true_vo, vx_true_vo, s=100, color='blue',
                                   marker='*', zorder=10, label='True VO')

                # Mark lines/points for MVP
                f.ax_joint.axhline(vx_true_mvp, color='tab:orange', linestyle='--',
                                   linewidth=1.5, alpha=0.7)
                f.ax_joint.axvline(vy_true_mvp, color='tab:orange', linestyle='--',
                                   linewidth=1.5, alpha=0.7)
                f.ax_joint.scatter(vy_true_mvp, vx_true_mvp, s=100, color='orange',
                                   marker='*', zorder=10, label='True MVP')

                # Rename legend labels for clarity
                handles, labels = f.ax_joint.get_legend_handles_labels()
                if labels:
                    labels[0] = "VO Samples"
                    if len(labels) > 1:
                        labels[1] = "MVP Samples"
                f.ax_joint.legend(handles=handles, labels=labels, loc='upper right')

                # ---------------------------------------------------------
                # CHANGES: Updated figure filename to include all fields
                # ---------------------------------------------------------
                fig_filename = (
                    f"{self.case_title_idx}_{self.source_of_uncertainty_idx}_"
                    f"{self.gs_int}_{dpsi_val}_{dcpa_val}.png"
                )
                fig_path = os.path.join(self.clustering.OUTPUT_DIR, fig_filename)
                f.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()  # or plt.close() if you don't want pop-up windows

        # ------------------------------------------------------------
        #  After ALL loops, create a DataFrame of our results
        # ------------------------------------------------------------
        if self.results:
            results_df = pd.DataFrame(self.results)
            datetime = strftime("%Y-%m-%d %H:%M:%S", gmtime())
            # You could also rename this file to include the case & source if you want
            results_csv_path = os.path.join(
                self.clustering.OUTPUT_DIR, 
                f"dataframe_results_{datetime}.csv"
            )
            results_df.to_csv(results_csv_path, index=False)
            print(f"\nDataframe logging saved to: {results_csv_path}")
        else:
            print("No conflicts detected across all scenarios. No CSV written.")


def main():
    """
    Orchestrates the entire conflict-resolution demonstration by:
      1. Selecting which uncertainties and sources are active
      2. Setting scenario noise parameters
      3. Running the main simulation loop (clustering + plotting + CSV logging).
    """
    # 1. Create a selector for user input
    selector = UncertaintyCaseSelector()

    # 2. Prompt user for the uncertainty type (pos, hdg, spd)
    case_title_selected, case_title_idx = selector.select_case_title()
    if case_title_idx is None:
        print("No valid case title selected. Exiting.")
        return

    # 3. Prompt user for the source of uncertainty (ownship, intruder, both)
    source_of_uncertainty, source_of_uncertainty_idx = selector.select_source_of_uncertainty()
    if source_of_uncertainty_idx is None:
        print("No valid source of uncertainty selected. Exiting.")
        return

    # 4. Initialize the conflict-resolution simulation
    sim = ConflictResolutionSimulation()
    sim.case_title_selected = case_title_selected
    sim.case_title_idx = case_title_idx
    sim.source_of_uncertainty = source_of_uncertainty
    sim.source_of_uncertainty_idx = source_of_uncertainty_idx

    # 5. Set the uncertainty switches
    sim.set_uncertainty_switches(case_title_idx, source_of_uncertainty_idx)

    # 6. Print chosen switches for clarity
    print(f"\nSelected case title: {sim.case_title_selected}")
    print(f"Selected source of uncertainty: {sim.source_of_uncertainty}")
    print("Switches:")
    print(f"  pos_uncertainty_on:  {sim.pos_uncertainty_on}")
    print(f"  hdg_uncertainty_on:  {sim.hdg_uncertainty_on}")
    print(f"  spd_uncertainty_on:  {sim.spd_uncertainty_on}")
    print(f"  src_ownship_on:      {sim.src_ownship_on}")
    print(f"  src_intruder_on:     {sim.src_intruder_on}\n")

    # 7. Set noise parameters based on the active switches
    sim.set_noise_parameters()

    # 8. Run the simulation
    sim.run_simulation()


if __name__ == "__main__":
    main()
