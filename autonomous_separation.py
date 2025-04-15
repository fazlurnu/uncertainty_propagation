import matplotlib.pyplot as plt
import numpy as np
from math import *

import seaborn as sns

import pandas as pd

import matplotlib.patches as patches
from matplotlib import pyplot, transforms
from matplotlib.transforms import Affine2D

from shapely.geometry import Point, LineString, Polygon
from shapely.affinity import translate, scale
from shapely.ops import nearest_points

def do_maneuver_check(gs_own, hdg_own, gs_int, hdg_int):      
    vn_int = gs_int * np.cos(np.radians(hdg_int))
    ve_int = gs_int * np.sin(np.radians(hdg_int))

    vn_int_local = vn_int * np.cos(np.radians(-hdg_own)) - ve_int * np.sin(np.radians(-hdg_own))
    ve_int_local = vn_int * np.sin(np.radians(-hdg_own)) + ve_int * np.cos(np.radians(-hdg_own))

    v_brg_A = np.degrees(np.arctan2(ve_int_local, vn_int_local))

    if(-70 <= v_brg_A <= 70):
        if(gs_own >= gs_int):
            return True
        else:
            return False
    elif(70 < v_brg_A <= 160):
            return False
    else:
        return True
        
def normalize_angle(angle):
        """ Normalize the angle to be within the range -180 to 180 degrees """
        while angle <= -180:
            angle += 360
        while angle > 180:
            angle -= 360
        return angle
    
def angle_difference(current, target):
    """ Calculate the smallest difference between two angles """
    diff = normalize_angle(target - current)
    if diff < 0:
        diff += 360
    return diff

def choose_right_vector(current, hdg_1, hdg_2, vector_1, vector_2):
    """ Choose the vector that is to the right of the current vector """
    diff1 = angle_difference(current, hdg_1)
    diff2 = angle_difference(current, hdg_2)

    if diff1 < diff2:
        return vector_1
    else:
        return vector_2
        
def VO(ownship_position, ownship_gs, ownship_trk,
       intruder_position, intruder_gs, intruder_trk,
       rpz, tlookahead, method = 0, scale = 1.0, resofach = 1.05):
    
    resofach = 1.05
    rpz = np.max(rpz * resofach)
    ownship_trk_rad = np.radians(ownship_trk)
    intruder_trk_rad = np.radians(intruder_trk)
    
    tp_1, tp_2 = get_cc_tp(ownship_position, intruder_position, rpz)
    
    ownship_velocity = Point(ownship_position.x + ownship_gs * cos(ownship_trk_rad), ownship_position.y + ownship_gs * sin(ownship_trk_rad))
    intruder_velocity = Point(intruder_gs * cos(intruder_trk_rad), intruder_gs * sin(intruder_trk_rad))
    intruder_velocity_on_intruder_pos = translate(intruder_velocity, xoff = intruder_position.x, yoff = intruder_position.y)
    
    if((tp_1 != None) & (tp_2 != None)):
        vo_0 = translate(ownship_position, xoff = intruder_velocity.x, yoff = intruder_velocity.y)
        vo_1 = translate(tp_1, xoff = intruder_velocity.x, yoff = intruder_velocity.y)
        vo_2 = translate(tp_2, xoff = intruder_velocity.x, yoff = intruder_velocity.y)

        vo_line_1 = LineString([vo_0, vo_1])
        vo_line_2 = LineString([vo_0, vo_2])

        # method = 0: opt, 1: spd change, 2: hdg change
        if(method == 0):
            cp_1 = nearest_points(vo_line_1, ownship_velocity)[0]
            cp_2 = nearest_points(vo_line_2, ownship_velocity)[0]

            cp_1_dist = cp_1.distance(ownship_velocity)
            cp_2_dist = cp_2.distance(ownship_velocity)
#             hdg_1 = np.arctan2(cp_1.y, cp_1.x)
#             hdg_2 = np.arctan2(cp_2.y, cp_2.x)

#             if(hdg_2 >= hdg_1):
#                 cp = cp_2
#             else:
#                 cp = cp_1
            if(cp_1_dist <= cp_2_dist):
                cp = cp_1
            else:
                cp = cp_2
                
        if(method == 1):
            if(ownship_gs > intruder_gs):
                ownship_velocity = Point(ownship_position.x + ownship_gs * scale * cos(ownship_trk_rad), ownship_position.y + ownship_gs * scale * sin(ownship_trk_rad))
            else:
                ownship_velocity = Point(ownship_position.x + ownship_gs / scale * cos(ownship_trk_rad), ownship_position.y + ownship_gs / scale * sin(ownship_trk_rad))
            
            cp_1 = nearest_points(vo_line_1, Point(ownship_velocity.x, ownship_velocity.y))[0]
            cp_2 = nearest_points(vo_line_2, Point(ownship_velocity.x, ownship_velocity.y))[0]

            cp_1_dist = cp_1.distance(ownship_velocity)
            cp_2_dist = cp_2.distance(ownship_velocity)
#             hdg_1 = np.arctan2(cp_1.y, cp_1.x)
#             hdg_2 = np.arctan2(cp_2.y, cp_2.x)

#             if(hdg_2 >= hdg_1):
#                 cp = cp_2
#             else:
#                 cp = cp_1
            if(cp_1_dist <= cp_2_dist):
                cp = cp_1
            else:
                cp = cp_2
                
        if(method == 2):
            scale = 0.95
            ownship_velocity = Point(ownship_position.x + ownship_gs * scale * cos(ownship_trk_rad), ownship_position.y + ownship_gs * scale * sin(ownship_trk_rad))
            cp_1 = nearest_points(vo_line_1, Point(ownship_velocity.x, ownship_velocity.y))[0]
            cp_2 = nearest_points(vo_line_2, Point(ownship_velocity.x, ownship_velocity.y))[0]

            cp_1_dist = cp_1.distance(ownship_velocity)
            cp_2_dist = cp_2.distance(ownship_velocity)
#             hdg_1 = np.arctan2(cp_1.y, cp_1.x)
#             hdg_2 = np.arctan2(cp_2.y, cp_2.x)

#             if(hdg_2 >= hdg_1):
#                 cp = cp_2
#             else:
#                 cp = cp_1
            if(cp_1_dist <= cp_2_dist):
                cp = cp_1
            else:
                cp = cp_2
                
        if(method == 4):
            cp_1 = nearest_points(vo_line_1, ownship_velocity)[0]
            cp_2 = nearest_points(vo_line_2, ownship_velocity)[0]

            curr_hdg = np.degrees(np.arctan2(ownship_velocity.y, ownship_velocity.x))
            hdg_1 = np.degrees(np.arctan2(cp_1.y, cp_1.x))
            hdg_2 = np.degrees(np.arctan2(cp_2.y, cp_2.x))

            cp = choose_right_vector(curr_hdg, hdg_1, hdg_2, cp_1, cp_2)

            gs_own = np.sqrt(ownship_velocity.x **2 + ownship_velocity.y **2)
            hdg_own = np.arctan2(ownship_velocity.y, ownship_velocity.x)

            gs_int = np.sqrt(intruder_velocity.x **2 + intruder_velocity.y **2)
            hdg_int = np.arctan2(intruder_velocity.y, intruder_velocity.x)

            do_maneuver = do_maneuver_check(gs_own, hdg_own, gs_int, hdg_int)

            if(do_maneuver):
                cp = cp
            else:
                cp = ownship_velocity
                
#         dx = ownship_position.x - intruder_position.x
#         dy = ownship_position.y - intruder_position.y
#         qdr = np.arctan2(dy, dx)

#         ## the priority based is here
#         qdr = np.degrees(qdr)

#         if(qdr >= 160 or qdr <= -160):
#             cp.x = 0
#             cp.y = 0

        return cp.x - ownship_position.x, cp.y - ownship_position.y
    else:
        return ownship_velocity.x - ownship_position.x, ownship_velocity.y - ownship_position.y
    
    

def create_point(start_point, length, angle_radians):
    # Calculate the new point's coordinates
    new_x = start_point.x + length * cos(angle_radians)
    new_y = start_point.y + length * sin(angle_radians)

    print(cos(angle_radians), sin(angle_radians), cos(angle_radians)**2 + sin(angle_radians)**2)

    # Create a new Point using Shapely
    new_point = Point(new_x, new_y)

    return new_point

def get_cc_tp(ownship_position, intruder_position, rpz):
    dx = intruder_position.x - ownship_position.x
    dy = intruder_position.y - ownship_position.y

    # print("Body frame: ", dx, dy)

    d = sqrt(dx**2 + dy**2)

    if(d > rpz):
        theta = atan2(dy, dx)
        beta = asin(rpz/d)
        side = sqrt(d**2 - rpz**2)

        tp_1_x = ownship_position.x + side * cos(theta - beta)
        tp_1_y = ownship_position.y + side * sin(theta - beta)
        tp_2_x = ownship_position.x + side * cos(theta + beta)
        tp_2_y = ownship_position.y + side * sin(theta + beta)

        return Point(tp_1_x, tp_1_y), Point(tp_2_x, tp_2_y)
    
    else:
        return None, None

def get_ip_line_circle(line, circle_center, circle_radius):
    # Create a Shapely Point for the circle center
    circle = Point(circle_center).buffer(circle_radius).boundary

    if line.intersects(circle):
        # Find the intersection points
        intersection_points = line.intersection(circle)

        if intersection_points.geom_type == 'Point':
            return intersection_points
        else:
            return [intersection_points.geoms[0], intersection_points.geoms[1]]
    else:
        return None

def get_ip_line_line(line1, line2, sfactor = 1000):
    s1 = scale(line1, xfact = sfactor, yfact = sfactor)
    s2 = scale(line2, xfact = sfactor, yfact = sfactor)

    intersection_point = s1.intersection(s2)

    if(intersection_point.is_empty):
        return None
    else:
        return intersection_point

def get_cp_line_point(line, point):
    x = np.array(point.coords[0])

    u = np.array(line.coords[0])
    v = np.array(line.coords[len(line.coords)-1])

    n = v - u
    n /= np.linalg.norm(n, 2)

    cp = u + n*np.dot(x - u, n)
    cp = Point(cp)

    return cp, cp.distance(point)

def get_speed_heading_band(ownship_position, ownship_gs, ownship_heading,
                           intruder_position, intruder_gs, intruder_heading):
    
    ownship_velocity = Point(ownship_gs * cos(ownship_heading), ownship_gs * sin(ownship_heading))
    intruder_velocity = Point(intruder_gs * cos(intruder_heading), intruder_gs * sin(intruder_heading))

    tp_1, tp_2 = get_cc_tp(ownship_position, intruder_position, rpz)

    vo_0 = translate(ownship_position, xoff = intruder_velocity.x, yoff = intruder_velocity.y)
    vo_1 = translate(tp_1, xoff = intruder_velocity.x, yoff = intruder_velocity.y)
    vo_2 = translate(tp_2, xoff = intruder_velocity.x, yoff = intruder_velocity.y)

    vo_line_1 = LineString([vo_0, vo_1])
    vo_line_2 = LineString([vo_0, vo_2])

    line = LineString([ownship_position, ownship_velocity])
    ip1 = get_ip_line_line(line, vo_line_1)
    ip2 = get_ip_line_line(line, vo_line_2)
    gs_1 = sqrt(ip1.x**2 + ip1.y**2)
    gs_2 = sqrt(ip2.x**2 + ip2.y**2)

    gs_band = [np.nan, np.nan]
    if(gs_1 < gs_2):
        gs_band = [gs_1, gs_2]
    else:
        gs_band = [gs_2, gs_1]

    ip1_heading = get_ip_line_circle(vo_line_1, ownship_position, ownship_gs)
    ip2_heading = get_ip_line_circle(vo_line_2, ownship_position, ownship_gs)

    heading_band = [np.nan, np.nan]
    if((ip1_heading != None) & (ip2_heading != None)):
        heading_1 = degrees(atan2(ip1_heading.y, ip1_heading.x))
        heading_2 = degrees(atan2(ip2_heading.y, ip2_heading.x))

        if(heading_1 < heading_2):
            heading_band = [heading_1, heading_2]
        else:
            heading_band = [heading_2, heading_1]

    return gs_band, heading_band

## plotting function (not necessary for conflict detection & resolution)
def plot_circle(ax, ownship_position, intruder_position, rpz):
    # Add the circle to the plot
    circle = plt.Circle((intruder_position.x, intruder_position.y), rpz, edgecolor='b', facecolor='none', label='Circle')

    # Add aircraft pos
    ax.plot(ownship_position.x, ownship_position.y, '-ro')
    ax.plot(intruder_position.x, intruder_position.y, '-bo')
    plot_line(ax, ownship_position, intruder_position)

    # Add the circle to the axis
    ax.add_patch(circle)

    # Set aspect ratio to be equal, so the circle looks like a circle
    ax.set_aspect('equal', adjustable='box')

    # Add labels and title
    plt.xlabel('X-body')
    plt.ylabel('Y-body')
    plt.title('Resolution Calculation')

def plot_line(ax, p1, p2, style = '--k'):
    ax.plot([p1.x, p2.x], [p1.y, p2.y], style)
    
def plot_tp(ax, tp_1, tp_2):
    # Add aircraft pos
    ax.plot(tp_1.x, tp_1.y, '-ko')
    ax.plot(tp_2.x, tp_2.y, '-ko')

def create_pos_noise_samples(x_ground_truth, y_ground_truth, nb_samples = 10000, sigma = 15):
    std_dev = sigma / 2.448
        
    cov = np.array([[std_dev**2, 0], 
                         [0, std_dev**2]])
            
    # Generate random samples from multivariate normal distribution
    x, y = np.random.multivariate_normal((0, 0), cov, nb_samples).T
    
    x_noise = x_ground_truth + x
    y_noise = y_ground_truth + y
    
    return x_noise, y_noise

def cre_conflict(xref, yref, trkref, gsref,
                 dpsi, dcpa, tlosh, spd, rpz = 50):
    
    trkref_rad = np.radians(trkref)
    
    trk = trkref + dpsi
    trk_rad = np.radians(trk)
    
    gsx = spd * np.cos(trk_rad)
    gsy = spd * np.sin(trk_rad)
    
    vrelx = gsref * np.cos(trkref_rad) - gsx
    vrely = gsref * np.sin(trkref_rad) - gsy
    
    vrel = np.sqrt(vrelx*vrelx + vrely*vrely)
    
    if(dcpa == 0):
        drelcpa = (tlosh*vrel + np.sqrt(rpz*rpz - dcpa*dcpa)) - (rpz * 0.01)
    else:
        drelcpa = tlosh*vrel + np.sqrt(rpz*rpz - dcpa*dcpa)

    dist = np.sqrt(drelcpa*drelcpa + dcpa*dcpa)
    
    # Rotation matrix diagonal and cross elements for distance vector
    rd      = drelcpa / dist
    rx      = dcpa / dist
    # Rotate relative velocity vector to obtain intruder bearing
    brn     = np.degrees(atan2(-rx * vrelx + rd * vrely,
                             rd * vrelx + rx * vrely))
    
    xint, yint = dist * np.cos(np.radians(brn)), dist * np.sin(np.radians(brn))

    return xint, yint, trk, spd

def cre_conflict_rand_dcpa(xref, yref, trkref, gsref,
                 dpsi, tlosh, spd, rpz = 50):
    
    trkref_rad = np.radians(trkref)
    
    trk = trkref + dpsi
    trk_rad = np.radians(trk)
    
    gsx = spd * np.cos(trk_rad)
    gsy = spd * np.sin(trk_rad)
    
    vrelx = gsref * np.cos(trkref_rad) - gsx
    vrely = gsref * np.sin(trkref_rad) - gsy
    
    vrel = np.sqrt(vrelx*vrelx + vrely*vrely)
    
    dcpa = np.random.uniform(0, rpz-1)
    
    drelcpa = tlosh*vrel + np.sqrt(rpz*rpz - dcpa*dcpa)
    
    dist = np.sqrt(drelcpa*drelcpa - dcpa*dcpa)
    
    # Rotation matrix diagonal and cross elements for distance vector
    rd      = drelcpa / dist
    rx      = dcpa / dist
    # Rotate relative velocity vector to obtain intruder bearing
    brn     = np.degrees(atan2(-rx * vrelx + rd * vrely,
                             rd * vrelx + rx * vrely))
    
    xint, yint = dist * np.cos(np.radians(brn)), dist * np.sin(np.radians(brn))
    
    return xint, yint, trk, spd

def conflict_detection_hor(ownship_position, ownship_gs, ownship_heading,
                       intruder_position, intruder_gs, intruder_heading,
                       rpz = 50, tlookahead = 15):
    
    ownship_heading_rad = np.radians(ownship_heading)
    intruder_heading_rad = np.radians(intruder_heading)
    
    ownship_velocity = Point(ownship_gs * cos(ownship_heading_rad), ownship_gs * sin(ownship_heading_rad))
    intruder_velocity = Point(intruder_gs * cos(intruder_heading_rad), intruder_gs * sin(intruder_heading_rad))

    dx = ownship_position.x - intruder_position.x
    dy = ownship_position.y - intruder_position.y
    dist = sqrt(dx**2 + dy**2)

    dvx = ownship_velocity.x - intruder_velocity.x
    dvy = ownship_velocity.y - intruder_velocity.y
    vrel = sqrt(dvx**2 + dvy**2)

    tcpa = -(dx * dvx + dy * dvy) / (dvx**2 + dvy**2)

    dcpa2 = (dist*dist - tcpa * tcpa * vrel * vrel)
    if(abs(dcpa2) < 1e-11):
        dcpa = 0
    else:
        dcpa = sqrt(dist*dist - tcpa * tcpa * vrel * vrel)

    LOS = dcpa < rpz

    if(LOS):
        tcrosshi = tcpa + sqrt(rpz*rpz - dcpa*dcpa)/vrel
        tcrosslo = tcpa - sqrt(rpz*rpz - dcpa*dcpa)/vrel

        tin = max(0.0, min(tcrosslo, tcrosshi))
        tout = max(tcrosslo, tcrosshi)

        is_in_conflict = (dcpa < rpz) & (tin < tlookahead)

        return dx, dy, tin, tout, dcpa, is_in_conflict
    else:
        ## if dcpa is more than rpz, return false
        tin = 0.0
        tout = 1e4
        return dx, dy, tin, tout, dcpa, LOS
    
def detect_conflict(row):
    dx, dy, tin, tout, dcpa, is_conflict = conflict_detection_hor(row['pos_ownship'], row['gs_own_noise'], row['hdg_own_noise'],
                                             row['pos_intruder'], row['gs_int_noise'], row['hdg_int_noise'])
    return dx, dy, tin, tout, dcpa, is_conflict

# Assuming conflict_detection_hor returns t1, t2, t3, t4
# and that df already contains the DataFrame with the necessary columns
def conf_reso_VO(row, rpz, tlookahead):
    vx, vy = VO(row['pos_ownship'], row['gs_own_noise'], row['hdg_own_noise'],
                row['pos_intruder'], row['gs_int_noise'], row['hdg_int_noise'],
                rpz = rpz, tlookahead=tlookahead)
    return vx, vy

def conf_reso_MVP(row, rpz, tlookahead):
    dcpa, vx, vy = MVP(row['pos_ownship'], row['gs_own_noise'], row['hdg_own_noise'],
                      row['pos_intruder'], row['gs_int_noise'], row['hdg_int_noise'],
                      rpz = rpz, tlookahead=tlookahead)
    return vx, vy

def conf_reso_MVP_int(row, rpz, tlookahead):
    dcpa, vx, vy = MVP(row['pos_intruder'], row['gs_int_noise'], row['hdg_int_noise'],
                      row['pos_ownship'], row['gs_own_noise'], row['hdg_own_noise'],
                      rpz = rpz, tlookahead=tlookahead)
    return vx, vy

def conf_reso_VO_int(row, rpz, tlookahead):
    vx, vy = VO(row['pos_intruder'], row['gs_int_noise'], row['hdg_int_noise'],
                row['pos_ownship'], row['gs_own_noise'], row['hdg_own_noise'],
                rpz = rpz, tlookahead=tlookahead)
    return vx, vy

def MVP(ownship_pos, ownship_gs, ownship_heading,
        intruder_pos, intruder_gs, intruder_heading,
        rpz, tlookahead = 15, resofach = 1.05):
    """Modified Voltage Potential (MVP) resolution method"""
    # Preliminary calculations-------------------------------------------------
    # Determine largest RPZ and HPZ of the conflict pair, use lookahead of ownship
    resofach = 1.05
    
    rpz_m = np.max(rpz * resofach)

    dtlook = tlookahead
    
    # Convert qdr from degrees to radians
    dx = ownship_pos.x - intruder_pos.x
    dy = ownship_pos.y - intruder_pos.y
    dist = sqrt(dx**2 + dy**2)
    qdr = np.arctan2(dy,dx)
    
    # Relative position vector between id1 and id2
    drel = np.array([np.sin(qdr) * dist, np.cos(qdr) * dist])

    # Write velocities as vectors and find relative velocity vector
    ownship_heading_rad = np.radians(ownship_heading)
    intruder_heading_rad = np.radians(intruder_heading)
    
    ownship_velocity = Point(ownship_gs * cos(ownship_heading_rad), ownship_gs * sin(ownship_heading_rad))
    intruder_velocity = Point(intruder_gs * cos(intruder_heading_rad), intruder_gs * sin(intruder_heading_rad))

    ## calc tcpa
    dvx = ownship_velocity.x - intruder_velocity.x
    dvy = ownship_velocity.y - intruder_velocity.y
    vrel2 = sqrt(dvx**2 + dvy**2)

    tcpa = -(dx * dvx + dy * dvy) / (dvx**2 + dvy**2)
    
    ## mvp again
    v1 = np.array([ownship_velocity.y, ownship_velocity.x])
    v2 = np.array([intruder_velocity.y, intruder_velocity.x])
    vrel = v1 - v2

    # Horizontal resolution----------------------------------------------------

    # Find horizontal distance at the tcpa (min horizontal distance)
    dcpa  = drel + vrel*tcpa
    dabsH = np.sqrt(dcpa[0] * dcpa[0] + dcpa[1] * dcpa[1])

#     print("dcpa mvp, ", dcpa)
#     print("drel mvp, ", drel)
    # Compute horizontal intrusion
    iH = rpz_m - dabsH

    # Exception handlers for head-on conflicts
    # This is done to prevent division by zero in the next step
    threshold = 0.001
#     threshold = 10
    
    if dabsH <= threshold:
        dabsH = threshold

        dcpa[0] = drel[1] / dist * dabsH
        dcpa[1] = -drel[0] / dist * dabsH

    # If intruder is outside the ownship PZ, then apply extra factor
    # to make sure that resolution does not graze IPZ
    if rpz_m < dist and dabsH < dist:
        # Compute the resolution velocity vector in horizontal direction.
        # abs(tcpa) because it bcomes negative during intrusion.
        erratum = np.cos(np.arcsin(rpz_m / dist)-np.arcsin(dabsH / dist))
        dv1 = ((rpz_m / erratum - dabsH) * dcpa[0]) / (abs(tcpa) * dabsH)
        dv2 = ((rpz_m / erratum - dabsH) * dcpa[1]) / (abs(tcpa) * dabsH)
    else:
        dv1 = (iH * dcpa[0]) / (abs(tcpa) * dabsH)
        dv2 = (iH * dcpa[1]) / (abs(tcpa) * dabsH)

#     print("here", dcpa, dabsH)
    # Combine resolutions------------------------------------------------------    
    # combine the dv components, since cooperative the dv can be halved
#     dv = np.array([dv1, dv2, 0])
    dv = np.array([dv1, dv2, 0])
    
    return dcpa, ownship_velocity.x + dv[1], ownship_velocity.y + dv[0]

