import numpy as np
from autonomous_separation import *

x_own, y_own = 0, 0
trk_own = 0
gs_own = 20

dcpa_start = 28
dcpa_end = 29

dpsi_start = 45
dpsi_end = 46

spd_start = 25
spd_end = 26

tlosh = 15

rpz = 50
dpsi = 180

sigma = 15

vy_init = gs_own * np.sin(np.radians(trk_own))
vx_init = gs_own * np.cos(np.radians(trk_own))

dcpa = 25
dpsi = 5
spd = 15

df_all = []

max_val = 1

gs_sigma = 3
hdg_sigma = 5

nb_samples = 5000

alpha_uncertainty = 0.4

for dpsi_here in range(dpsi, dpsi+2, 2):
    df = []
    
    x_int, y_int, trk_int, gs_int = cre_conflict(x_own, y_own, trk_own, gs_own, dpsi_here, dcpa, tlosh, spd, rpz)
    
    ownship_pos = Point(x_own, y_own)
    intruder_pos = Point(x_int, y_int)

    x_o, y_o = create_pos_noise_samples(x_own, y_own, nb_samples=nb_samples, sigma = 0)
    x_i, y_i = create_pos_noise_samples(x_int, y_int, nb_samples=nb_samples, sigma = 0)

    hdg_ownship = np.random.normal(trk_own, 0, nb_samples)
    gs_ownship = np.random.normal(gs_own, 0, nb_samples)
    hdg_intruder = np.random.normal(trk_int, hdg_sigma, nb_samples)
    gs_intruder = np.random.normal(gs_int, 0, nb_samples)

    df = pd.DataFrame({
                      'x_own_true': x_own, 'y_own_true': y_own,
                      'x_own_noise': x_o, 'y_own_noise': y_o,
                      'hdg_own': hdg_ownship, 'gs_own': gs_ownship,
                      'x_int_true': x_int, 'y_int_true': y_int,
                      'x_int_noise': x_i, 'y_int_noise': y_i,
                      'hdg_int': hdg_intruder, 'gs_int': gs_intruder})

    # Create Shapely Point objects from 'x_own_noise' and 'y_own_noise'
    df['pos_ownship'] = [Point(x, y) for x, y in zip(df['x_own_noise'], df['y_own_noise'])]
    df['pos_intruder'] = [Point(x, y) for x, y in zip(df['x_int_noise'], df['y_int_noise'])]

    # Define a function to apply conflict_detection_hor to each row

    # Apply the function to each row and unpack the tuple into separate columns
    df[['dx', 'dy', 'tin', 'tout', 'dcpa', 'is_conflict']] = df.apply(lambda row: pd.Series(detect_conflict(row)), axis=1)

    # Apply the function to each row and unpack the tuple into separate columns
    df[['vx_vo', 'vy_vo']] = df.apply(lambda row: pd.Series(conf_reso_VO(row)), axis=1)
    df[['vx_mvp', 'vy_mvp']] = df.apply(lambda row: pd.Series(conf_reso_MVP(row)), axis=1)
    
#     df_all.append(df)
    
#     df = pd.concat(df_all, ignore_index=True, sort=False)

    df = df.loc[df['is_conflict']]

    df_vo = df[['vx_vo', 'vy_vo']].copy()
    df_vo['vx'] = df['vx_vo']
    df_vo['vy'] = df['vy_vo']
    df_vo['Conf Reso'] = 'VO'

    df_mvp = df[['vx_mvp', 'vy_mvp']].copy()
    df_mvp['vx'] = df['vx_mvp']
    df_mvp['vy'] = df['vy_mvp']
    df_mvp['Conf Reso'] = 'MVP'

    # df_plot = pd.concat([df_vo, df_svo], ignore_index=True, sort=False)
    df_plot = pd.concat([df_vo, df_mvp], ignore_index=True, sort=False)

    f = sns.jointplot(
        data=df_plot,
        x="vy", y="vx", hue="Conf Reso",
        kind="scatter", alpha = alpha_uncertainty, label = None, palette = ['tab:blue', 'tab:orange', 'tab:red', 'tab:purple']
    )

    f.set_axis_labels("Resolution $V_{y-body}$ [kts]", "Resolution $V_{x-body}$ [kts]")

    # Setting x and y limits
    max_offset_vy = 10
    max_offset_vx = 10

    f.ax_joint.set_xlim(vy_init - max_offset_vy, vy_init + max_offset_vy)  # set x limits
    f.ax_joint.set_ylim(vx_init - max_offset_vx, vx_init + max_offset_vx)  # set y limits

    tp_1, tp_2 = get_cc_tp(Point(x_own, y_own), Point(x_int, y_int), rpz)

    int_vel = Point(gs_int * np.cos(np.radians(trk_int)), gs_int * np.sin(np.radians(trk_int)))
    ownship_pos = Point(x_own, y_own)

    if((tp_1 != None) & (tp_2 != None)):
    # if(False):

        vo_0 = translate(ownship_pos, xoff = int_vel.x, yoff = int_vel.y)
        vo_1 = translate(tp_1, xoff = int_vel.x, yoff = int_vel.y)
        vo_2 = translate(tp_2, xoff = int_vel.x, yoff = int_vel.y)

        ## for plotting, extend the line
        dx_1 = vo_1.x - vo_0.x
        dy_1 = vo_1.y - vo_0.y

        ## for plotting, extend the line
        dx_2 = vo_2.x - vo_0.x
        dy_2 = vo_2.y - vo_0.y

        # Define the extension length
        extension_length = 100

        # Calculate the endpoint of the extended line
        extended_endpoint_1 = Point(vo_1.x + dx_1 * extension_length, vo_1.y + dy_1 * extension_length)
        extended_endpoint_2 = Point(vo_2.x + dx_2 * extension_length, vo_2.y + dy_2 * extension_length)

        vo_line_1 = LineString([vo_0, vo_1])
        vo_line_2 = LineString([vo_0, vo_2])

        scale_factor = 1

        f.ax_joint.plot([vo_0.y, extended_endpoint_1.y], [vo_0.x, extended_endpoint_1.x], color = 'k');
        f.ax_joint.plot([vo_0.y, extended_endpoint_2.y], [vo_0.x, extended_endpoint_2.x], color = 'k');

        ## Draw between

        # Extract coordinates from the lines
        coords_1 = list(vo_line_1.coords)
        coords_2 = list(vo_line_2.coords)

        # Combine the coordinates to form a polygon
        # Note: The second line's coordinates need to be reversed to properly form the polygon
        polygon_coords = coords_1 + coords_2[::-1]

        # Create the polygon
        polygon = Polygon(polygon_coords)

        ax = f.ax_joint
        x_poly, y_poly = polygon.exterior.xy
        ax.fill(y_poly, x_poly, alpha=0.3, fc='red', zorder = -1);

    vx_true_vo, vy_true_vo = VO(Point(x_own, y_own), gs_own, trk_own,
                   Point(x_int, y_int), gs_int, trk_int,
                   50, method = 0)

    vx_true_svo, vy_true_svo = VO(Point(x_own, y_own), gs_own, trk_own,
                   Point(x_int, y_int), gs_int, trk_int,
                   50, method = 4)

    vx_true_125, vy_true_125 = VO(Point(x_own, y_own), gs_own, trk_own,
                   Point(x_int, y_int), gs_int, trk_int,
                   50, method = 1)

    vx_true_95, vy_true_95 = VO(Point(x_own, y_own), gs_own, trk_own,
                   Point(x_int, y_int), gs_int, trk_int,
                   50, method = 2)

    _, vx_true_mvp, vy_true_mvp = MVP(Point(x_own, y_own), gs_own, trk_own,
                       Point(x_int, y_int), gs_int, trk_int,
                       50)

    handles, labels = f.ax_joint.get_legend_handles_labels()
    f.ax_joint.legend(handles=handles, labels=labels, loc='upper right', bbox_to_anchor=(1.25, 1.2))

    f.ax_joint.axhline(vx_true_vo, color='tab:blue', linestyle='--', linewidth=1.5, alpha = 0.7)
    f.ax_joint.axvline(vy_true_vo, color='tab:blue', linestyle='--', linewidth=1.5, alpha = 0.7)
    f.ax_joint.scatter(vy_true_vo, vx_true_vo, s = 100, color = 'blue', marker = '*', zorder = 10, label = 'True VO', )

    f.ax_joint.axhline(vx_true_mvp, color='tab:orange', linestyle='--', linewidth=1.5, alpha = 0.7)
    f.ax_joint.axvline(vy_true_mvp, color='tab:orange', linestyle='--', linewidth=1.5, alpha = 0.7)
    f.ax_joint.scatter(vy_true_mvp, vx_true_mvp, s = 100, color = 'orange', marker = '*', zorder = 10, label = 'True MVP')

    handles, labels = f.ax_joint.get_legend_handles_labels()
    # Remove the first legend item
    # handles = handles[1:]
    # labels = labels[1:]
    labels[0] = "VO Samples"
    labels[1] = "MVP Samples"

    # f.ax_joint.legend(handles=handles, labels=labels, loc='upper right', bbox_to_anchor=(1.45, 1.25))
    f.ax_joint.legend(handles=handles, labels=labels, loc='upper right')
    
    # At the end of your plotting code
    if(spd == 15 and dpsi_here == 5):
        print(round(df_vo['vx'].mean(), 2), round(df_vo['vx'].std(),2))
        print(round(df_mvp['vx'].mean(), 2), round(df_mvp['vx'].std(), 2))
        
        print(round(df_vo['vy'].mean(),2), round(df_vo['vy'].std(),2))
        print(round(df_mvp['vy'].mean(),2), round(df_mvp['vy'].std(),2))
    
    # f.savefig(f'output_figure_{spd}_{dpsi_here}_{dcpa}.png', dpi=300, bbox_inches='tight')
    # plt.close(f.fig)  # Close the figure to free up memory
    # df.to_csv(f'output_dist_{spd}_{dpsi_here}_{dcpa}.csv', index = False)
    plt.show()
