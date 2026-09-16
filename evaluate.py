import numpy as np
import pandas as pd

def evaluate_predictions(df_true, df_pred, dt=0.1,
                         rv_th=2.0, rh_th=5.0, ra_th=1.0, rj_th=2.0, ttc_th=3.0):
    """
    Computes local IEEE ITSS scores (Accuracy, Safety, Comfort, Total)
    given ground truth and predicted FAV trajectories.

    FIXED: derivatives now use plain forward differences (np.diff without
    prepend), matching the published formulas exactly. The previous version
    used prepend=x[0], which forces the first velocity value to 0 regardless
    of the real speed, then amplifies that artifact into a huge spurious
    acceleration/jerk spike on every segment -- this was silently capping
    Comfort near-zero no matter how smooth the actual predicted trajectory
    was.
    """
    total_scores, accuracy_scores, safety_scores, comfort_scores = [], [], [], []

    unique_segments = df_pred['Segment_ID'].unique()

    for seg_id in unique_segments:
        seg_true = df_true[df_true['Segment_ID'] == seg_id].sort_values('Time_Index')
        seg_pred = df_pred[df_pred['Segment_ID'] == seg_id].sort_values('Time_Index')

        x_true = seg_true['Pos_FAV'].values
        x_lv = seg_true['Pos_LV'].values
        x_pred = seg_pred['Pos_FAV'].values

        # 1. Derived quantities via plain forward differences (no artificial padding).
        # Each derivative level is one element shorter than the one before it --
        # this matches the challenge's own notation (v defined on t=1..T-1, etc.)
        v_true = np.diff(x_true) / dt   # length N-1
        v_pred = np.diff(x_pred) / dt   # length N-1
        v_lv   = np.diff(x_lv) / dt     # length N-1

        a_true = np.diff(v_true) / dt   # length N-2
        a_pred = np.diff(v_pred) / dt   # length N-2

        j_pred = np.diff(a_pred) / dt   # length N-3 (predicted jerk only, per spec)

        # Headway / spatial gap are position-level quantities -- no differencing,
        # so these were never affected by the bug.
        h_true = x_lv - x_true          # length N
        h_pred = x_lv - x_pred          # length N
        g_pred = h_pred - 4.5           # length N

        # --- ACCURACY SCORE ---
        rv = np.sqrt(np.mean((v_pred - v_true) ** 2))
        rh = np.sqrt(np.mean((h_pred - h_true) ** 2))
        ra = np.sqrt(np.mean((a_pred - a_true) ** 2))

        s_speed = max(0.0, 1.0 - (rv / rv_th))
        s_headway = max(0.0, 1.0 - (rh / rh_th))
        s_acc = max(0.0, 1.0 - (ra / ra_th))

        s_a = 0.4 * s_speed + 0.4 * s_headway + 0.2 * s_acc

        # --- SAFETY SCORE ---
        # Hard failure: spatial gap < 0 at ANY submitted timestep (full-length check)
        if np.any(g_pred < 0):
            s_s = 0.0
        else:
            rel_speed = v_pred - v_lv          # length N-1
            g_for_ttc = g_pred[:-1]            # align to N-1 to match rel_speed
            ttc = np.where(rel_speed > 0, g_for_ttc / np.maximum(rel_speed, 1e-5), np.inf)
            ttc_violations = np.sum(ttc < ttc_th) / len(ttc)
            s_s = 1.0 - ttc_violations

        # --- COMFORT SCORE ---
        rj = np.sqrt(np.mean(j_pred ** 2))
        s_c = max(0.0, 1.0 - (rj / rj_th))

        # --- TOTAL SEGMENT SCORE ---
        s_total = 0.5 * s_a + 0.3 * s_s + 0.2 * s_c

        accuracy_scores.append(s_a)
        safety_scores.append(s_s)
        comfort_scores.append(s_c)
        total_scores.append(s_total)

    print(f"--- Local Evaluation Results ---")
    print(f"Accuracy Score : {np.mean(accuracy_scores):.4f}")
    print(f"Safety Score   : {np.mean(safety_scores):.4f}")
    print(f"Comfort Score  : {np.mean(comfort_scores):.4f}")
    print(f"Final Score    : {np.mean(total_scores):.4f}")

    return np.mean(total_scores)