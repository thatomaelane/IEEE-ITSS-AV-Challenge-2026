# ============================================================================
# COMPLETE PIPELINE – TOP‑15 SUBMISSION-Advocate-Model (single cell)
# ============================================================================
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt

# ============================================================================
# 1. PATHS & SETTINGS
# ============================================================================
TRAIN_PATH = "/kaggle/input/datasets/thatomaelane/ieee-comp/train.csv"
TEST_PATH  = "/kaggle/input/datasets/thatomaelane/ieee-comp/test.csv"

TRAIN = True                # set to False to skip training (if checkpoints exist)
SMOOTHING_CUTOFF = 1.0      # best from validation (1.0 Hz)
N_ENSEMBLE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
dt = 0.1
HIST_LEN = 100
HORIZON = 200
HIDDEN = 384
ATTN_DIM = 64
DROPOUT = 0.15
JERK_WEIGHT = 0.05
BATCH_SIZE = 256
N_EPOCHS = 150

# ============================================================================
# 2. EVALUATION FUNCTION (matches official metric)
# ============================================================================
def evaluate_predictions(df_true, df_pred, dt=0.1,
                         rv_th=2.0, rh_th=5.0, ra_th=1.0, rj_th=2.0, ttc_th=3.0):
    total_scores, accuracy_scores, safety_scores, comfort_scores = [], [], [], []
    unique_segments = df_pred['Segment_ID'].unique()
    for seg_id in unique_segments:
        seg_true = df_true[df_true['Segment_ID'] == seg_id].sort_values('Time_Index')
        seg_pred = df_pred[df_pred['Segment_ID'] == seg_id].sort_values('Time_Index')
        x_true = seg_true['Pos_FAV'].values
        x_lv   = seg_true['Pos_LV'].values
        x_pred = seg_pred['Pos_FAV'].values
        v_true = np.diff(x_true) / dt
        v_pred = np.diff(x_pred) / dt
        v_lv   = np.diff(x_lv) / dt
        a_true = np.diff(v_true) / dt
        a_pred = np.diff(v_pred) / dt
        j_pred = np.diff(a_pred) / dt
        h_true = x_lv - x_true
        h_pred = x_lv - x_pred
        g_pred = h_pred - 4.5
        rv = np.sqrt(np.mean((v_pred - v_true) ** 2))
        rh = np.sqrt(np.mean((h_pred - h_true) ** 2))
        ra = np.sqrt(np.mean((a_pred - a_true) ** 2))
        s_speed = max(0.0, 1.0 - (rv / rv_th))
        s_headway = max(0.0, 1.0 - (rh / rh_th))
        s_acc = max(0.0, 1.0 - (ra / ra_th))
        s_a = 0.4 * s_speed + 0.4 * s_headway + 0.2 * s_acc
        if np.any(g_pred < 0):
            s_s = 0.0
        else:
            rel_speed = v_pred - v_lv
            g_for_ttc = g_pred[:-1]
            ttc = np.where(rel_speed > 0, g_for_ttc / np.maximum(rel_speed, 1e-5), np.inf)
            ttc_violations = np.sum(ttc < ttc_th) / len(ttc)
            s_s = 1.0 - ttc_violations
        rj = np.sqrt(np.mean(j_pred ** 2))
        s_c = max(0.0, 1.0 - (rj / rj_th))
        s_total = 0.5 * s_a + 0.3 * s_s + 0.2 * s_c
        accuracy_scores.append(s_a); safety_scores.append(s_s)
        comfort_scores.append(s_c); total_scores.append(s_total)
    if len(total_scores) == 0:
        return 0.0
    print(f"--- Local Evaluation Results ---")
    print(f"Accuracy Score : {np.mean(accuracy_scores):.4f}")
    print(f"Safety Score   : {np.mean(safety_scores):.4f}")
    print(f"Comfort Score  : {np.mean(comfort_scores):.4f}")
    print(f"Final Score    : {np.mean(total_scores):.4f}")
    return np.mean(total_scores)

# ============================================================================
# 3. HELPER FUNCTIONS
# ============================================================================
def compute_idm_acc(v_fav, v_lv, gap, a_max=1.4, b_comf=1.8, v0=30.0, s0=2.0, T_headway=1.2, delta=4.0):
    s_gap = np.maximum(0.1, gap)
    delta_v = v_fav - v_lv
    s_star = s0 + np.maximum(0.0, v_fav * T_headway) + (v_fav * delta_v) / (2.0 * np.sqrt(a_max * b_comf) + 1e-6)
    s_star = np.maximum(s0, s_star)
    a_idm = a_max * (1.0 - (v_fav / v0)**delta - (s_star / s_gap)**2)
    return np.clip(a_idm, -4.0, 2.0)

def build_segment_tensors(seg_ids, df):
    hist_feats, fut_lv_feats, fut_true_pos, init_state, seg_id_list = [], [], [], [], []
    for seg_id in seg_ids:
        seg = df[df['Segment_ID'] == seg_id].sort_values('Time_Index')
        hist = seg[seg['Time_Index'] < 10.00]
        fut  = seg[seg['Time_Index'] >= 10.00]
        if len(hist) != HIST_LEN or len(fut) != HORIZON:
            continue
        h_gap  = (hist['Pos_LV'] - hist['Pos_FAV'] - 4.5).values
        h_rel_v = (hist['Speed_LV'] - hist['Speed_FAV']).values
        h_rel_a = (hist['Acc_LV'] - hist['Acc_FAV']).values
        h_type  = hist['Type_LV'].values.astype(np.float32)
        h_id    = hist['ID_LV'].values.astype(np.float32) / 13.0
        h_headway = hist['Spatial_Headway'].values.astype(np.float32)
        h_ttc = np.where(h_rel_v > 0, h_gap / np.maximum(h_rel_v, 0.01), 100.0)
        h_ttc = np.clip(h_ttc, 0, 100)
        h_idm = compute_idm_acc(hist['Speed_FAV'].values, hist['Speed_LV'].values, h_gap)
        h_feat = np.stack([hist['Speed_FAV'].values, hist['Speed_LV'].values,
                           h_gap, h_rel_v, h_rel_a, hist['Acc_LV'].values,
                           h_type, h_id, h_headway, h_ttc, h_idm], axis=-1)
        pos0 = hist['Pos_FAV'].values[-1]
        f_lv_rel = np.stack([fut['Speed_LV'].values, fut['Acc_LV'].values, fut['Pos_LV'].values - pos0], axis=-1)
        f_lv_raw = np.stack([fut['Speed_LV'].values, fut['Acc_LV'].values, fut['Pos_LV'].values], axis=-1)
        f_true_pos = fut['Pos_FAV'].values
        v0 = max(0.0, (hist['Pos_FAV'].values[-1] - hist['Pos_FAV'].values[-2]) / dt)
        hist_feats.append(h_feat)
        fut_lv_feats.append(f_lv_rel)
        fut_true_pos.append(f_true_pos)
        init_state.append([pos0, v0])
        seg_id_list.append(seg_id)
    return dict(hist=np.array(hist_feats, dtype=np.float32), fut_lv=np.array(fut_lv_feats, dtype=np.float32),
                fut_true_pos=np.array(fut_true_pos, dtype=np.float32),
                init_state=np.array(init_state, dtype=np.float32), seg_ids=seg_id_list)

def build_test_tensors(df):
    seg_ids = df['Segment_ID'].unique()
    hist_list, fut_lv_list, fut_lv_raw_list, init_list, seg_id_list = [], [], [], [], []
    for seg_id in seg_ids:
        seg = df[df['Segment_ID'] == seg_id].sort_values('Time_Index')
        hist = seg[seg['Time_Index'] < 10.00]
        fut = seg[seg['Time_Index'] >= 10.00]
        if len(hist) != HIST_LEN or len(fut) != HORIZON:
            continue
        h_gap = (hist['Pos_LV'] - hist['Pos_FAV'] - 4.5).values
        h_rel_v = (hist['Speed_LV'] - hist['Speed_FAV']).values
        h_rel_a = (hist['Acc_LV'] - hist['Acc_FAV']).values
        h_type = hist['Type_LV'].values.astype(np.float32)
        h_id = hist['ID_LV'].values.astype(np.float32) / 13.0
        h_headway = hist['Spatial_Headway'].values.astype(np.float32)
        h_ttc = np.where(h_rel_v > 0, h_gap / np.maximum(h_rel_v, 0.01), 100.0)
        h_ttc = np.clip(h_ttc, 0, 100)
        h_idm = compute_idm_acc(hist['Speed_FAV'].values, hist['Speed_LV'].values, h_gap)
        h_feat = np.stack([hist['Speed_FAV'].values, hist['Speed_LV'].values,
                           h_gap, h_rel_v, h_rel_a, hist['Acc_LV'].values,
                           h_type, h_id, h_headway, h_ttc, h_idm], axis=-1)
        pos0 = hist['Pos_FAV'].values[-1]
        f_lv_rel = np.stack([fut['Speed_LV'].values, fut['Acc_LV'].values, fut['Pos_LV'].values - pos0], axis=-1)
        f_lv_raw = np.stack([fut['Speed_LV'].values, fut['Acc_LV'].values, fut['Pos_LV'].values], axis=-1)
        v0 = max(0.0, (hist['Pos_FAV'].values[-1] - hist['Pos_FAV'].values[-2]) / dt)
        hist_list.append(h_feat)
        fut_lv_list.append(f_lv_rel)
        fut_lv_raw_list.append(f_lv_raw)
        init_list.append([pos0, v0])
        seg_id_list.append(seg_id)
    return dict(hist=np.array(hist_list, dtype=np.float32),
                fut_lv=np.array(fut_lv_list, dtype=np.float32),
                fut_lv_raw=np.array(fut_lv_raw_list, dtype=np.float32),
                init_state=np.array(init_list, dtype=np.float32),
                seg_ids=seg_id_list)

def apply_lowpass(accel_seq, cutoff_hz, order=4):
    nyq = 0.5 / dt
    b, a = butter(order, cutoff_hz / nyq, btype='low')
    padlen = 3 * max(len(a), len(b))
    if len(accel_seq) <= padlen:
        return accel_seq
    return filtfilt(b, a, accel_seq)

def integrate(pos0, v0, accel_seq):
    pos, v = pos0, v0
    positions = []
    for a in accel_seq:
        pos = pos + v * dt + 0.5 * a * (dt ** 2)
        v = max(0.0, v + a * dt)
        positions.append(pos)
    return positions

# ============================================================================
# 4. MODEL DEFINITION
# ============================================================================
class EnhancedLSTM(nn.Module):
    def __init__(self, feat_dim=11, hidden=HIDDEN, attn_dim=ATTN_DIM, dropout=DROPOUT):
        super().__init__()
        self.encoder = nn.LSTM(input_size=feat_dim, hidden_size=hidden, batch_first=True, num_layers=1)
        self.future_encoder = nn.LSTM(input_size=3, hidden_size=hidden, batch_first=True, bidirectional=True)
        self.query_proj = nn.Linear(hidden, attn_dim)
        self.key_proj   = nn.Linear(hidden * 2, attn_dim)
        self.value_proj = nn.Linear(hidden * 2, attn_dim)
        self.decoder_cell = nn.LSTMCell(input_size=6 + attn_dim, hidden_size=hidden)
        self.out = nn.Sequential(nn.Linear(hidden, 128), nn.ReLU(), nn.Dropout(dropout), nn.Linear(128, 1))

    def forward(self, hist, fut_lv, fut_lv_raw, init_state, hist_mean_t, hist_std_t,
                fut_true_pos=None, teacher_forcing_ratio=0.0):
        B = hist.shape[0]
        _, (h, c) = self.encoder(hist)
        h, c = h.squeeze(0), c.squeeze(0)
        future_enc, _ = self.future_encoder(fut_lv)
        keys   = self.key_proj(future_enc)
        values = self.value_proj(future_enc)
        pos = init_state[:, 0].clone()
        vel = init_state[:, 1].clone()
        acc = torch.zeros(B, device=hist.device)
        positions = []
        for t in range(HORIZON):
            v_lv_t, a_lv_t, pos_lv_t = fut_lv_raw[:, t, 0], fut_lv_raw[:, t, 1], fut_lv_raw[:, t, 2]
            gap = pos_lv_t - pos - 4.5
            rel_v = v_lv_t - vel
            query = self.query_proj(h).unsqueeze(1)
            scores = torch.bmm(query, keys.transpose(1, 2)) / (ATTN_DIM ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1)
            context = torch.bmm(attn_weights, values).squeeze(1)
            step_in = torch.cat([torch.stack([
                (vel - hist_mean_t[0]) / hist_std_t[0],
                (v_lv_t - hist_mean_t[1]) / hist_std_t[1],
                (gap - hist_mean_t[2]) / hist_std_t[2],
                (rel_v - hist_mean_t[3]) / hist_std_t[3],
                (a_lv_t - hist_mean_t[4]) / hist_std_t[4],
                acc / 4.0,
            ], dim=-1), context], dim=-1)
            h, c = self.decoder_cell(step_in, (h, c))
            da = torch.tanh(self.out(h).squeeze(-1)) * 4.0
            acc = 0.25 * da + 0.75 * acc
            new_pos = pos + vel * dt + 0.5 * acc * (dt ** 2)
            new_vel = torch.clamp(vel + acc * dt, min=0.0)
            positions.append(new_pos)
            if self.training and fut_true_pos is not None and teacher_forcing_ratio > 0:
                use_tf = (torch.rand(B, device=hist.device) < teacher_forcing_ratio).float()
                true_pos_t = fut_true_pos[:, t]
                true_vel_t = (fut_true_pos[:, 0] - init_state[:, 0]) / dt if t == 0 else \
                             (fut_true_pos[:, t] - fut_true_pos[:, t-1]) / dt
                pos = use_tf * true_pos_t + (1 - use_tf) * new_pos
                vel = use_tf * true_vel_t + (1 - use_tf) * new_vel
            else:
                pos, vel = new_pos, new_vel
        return torch.stack(positions, dim=1)

def forward_with_accel(model, hist, fut_lv, fut_lv_raw, init_state, hist_mean_t, hist_std_t):
    B = hist.shape[0]
    _, (h, c) = model.encoder(hist)
    h, c = h.squeeze(0), c.squeeze(0)
    future_enc, _ = model.future_encoder(fut_lv)
    keys = model.key_proj(future_enc)
    values = model.value_proj(future_enc)
    pos = init_state[:, 0].clone()
    vel = init_state[:, 1].clone()
    acc = torch.zeros(B, device=hist.device)
    accels = []
    for t in range(HORIZON):
        v_lv_t, a_lv_t, pos_lv_t = fut_lv_raw[:, t, 0], fut_lv_raw[:, t, 1], fut_lv_raw[:, t, 2]
        gap = pos_lv_t - pos - 4.5
        rel_v = v_lv_t - vel
        query = model.query_proj(h).unsqueeze(1)
        scores = torch.bmm(query, keys.transpose(1, 2)) / (ATTN_DIM ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, values).squeeze(1)
        step_in = torch.cat([torch.stack([
            (vel - hist_mean_t[0]) / hist_std_t[0],
            (v_lv_t - hist_mean_t[1]) / hist_std_t[1],
            (gap - hist_mean_t[2]) / hist_std_t[2],
            (rel_v - hist_mean_t[3]) / hist_std_t[3],
            (a_lv_t - hist_mean_t[4]) / hist_std_t[4],
            acc / 4.0,
        ], dim=-1), context], dim=-1)
        h, c = model.decoder_cell(step_in, (h, c))
        da = torch.tanh(model.out(h).squeeze(-1)) * 4.0
        acc = 0.25 * da + 0.75 * acc
        new_pos = pos + vel * dt + 0.5 * acc * (dt ** 2)
        new_vel = torch.clamp(vel + acc * dt, min=0.0)
        accels.append(acc.clone())
        pos, vel = new_pos, new_vel
    return torch.stack(accels, dim=1)

# ============================================================================
# 5. TRAINING (if TRAIN=True)
# ============================================================================
if TRAIN:
    print("--- Loading data for training ---")
    full_df = pd.read_csv(TRAIN_PATH)
    unique_segments = full_df['Segment_ID'].unique()
    np.random.seed(42)
    val_segments = set(np.random.choice(unique_segments, size=int(len(unique_segments) * 0.03), replace=False))
    train_segments = [s for s in unique_segments if s not in val_segments]
    val_segments = list(val_segments)

    train_data = build_segment_tensors(train_segments, full_df)
    val_data   = build_segment_tensors(val_segments, full_df)

    hist_mean = train_data['hist'].reshape(-1, 11).mean(axis=0)
    hist_std  = train_data['hist'].reshape(-1, 11).std(axis=0) + 1e-6
    lv_mean   = train_data['fut_lv'].reshape(-1, 3).mean(axis=0)
    lv_std    = train_data['fut_lv'].reshape(-1, 3).std(axis=0) + 1e-6

    hist_mean_t = torch.tensor(hist_mean, device=device)
    hist_std_t  = torch.tensor(hist_std, device=device)

    train_hist = torch.tensor((train_data['hist'] - hist_mean) / hist_std, device=device)
    train_fut_lv = torch.tensor((train_data['fut_lv'] - lv_mean) / lv_std, device=device)
    train_fut_lv_raw = torch.tensor(np.stack([
        train_data['fut_lv'][:, :, 0], train_data['fut_lv'][:, :, 1],
        train_data['fut_lv'][:, :, 2] + train_data['init_state'][:, 0:1]
    ], axis=-1), device=device, dtype=torch.float32)
    train_true_pos = torch.tensor(train_data['fut_true_pos'], device=device)
    train_init = torch.tensor(train_data['init_state'], device=device)

    val_hist = torch.tensor((val_data['hist'] - hist_mean) / hist_std, device=device)
    val_fut_lv = torch.tensor((val_data['fut_lv'] - lv_mean) / lv_std, device=device)
    val_fut_lv_raw = torch.tensor(np.stack([
        val_data['fut_lv'][:, :, 0], val_data['fut_lv'][:, :, 1],
        val_data['fut_lv'][:, :, 2] + val_data['init_state'][:, 0:1]
    ], axis=-1), device=device, dtype=torch.float32)
    val_init = torch.tensor(val_data['init_state'], device=device)

    df_val_horizon = full_df[full_df['Segment_ID'].isin(val_data['seg_ids']) & (full_df['Time_Index'] >= 10.00)].copy()

    # --- Train 5‑member ensemble ---
    print("--- Training ensemble ---")
    N_ENSEMBLE = 5
    for member_idx in range(N_ENSEMBLE):
        print(f"\nTraining member {member_idx+1}/{N_ENSEMBLE}")
        torch.manual_seed(member_idx * 1000 + 42)
        np.random.seed(member_idx * 1000 + 42)

        m = EnhancedLSTM().to(device)
        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        N_train = train_hist.shape[0]
        steps_per_epoch = (N_train + BATCH_SIZE - 1) // BATCH_SIZE
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS * steps_per_epoch, eta_min=1e-6)
        best_m_score = -1.0
        epochs_since_best = 0

        for epoch in range(N_EPOCHS):
            m.train()
            tf_ratio = max(0.0, 1.0 - epoch / (N_EPOCHS * 0.85))
            perm = torch.randperm(N_train, device=device)
            for i in range(0, N_train, BATCH_SIZE):
                idx = perm[i:i+BATCH_SIZE]
                h_b = train_hist[idx]; lv_b = train_fut_lv[idx]; lv_raw_b = train_fut_lv_raw[idx]
                tp_b = train_true_pos[idx]; init_b = train_init[idx]
                with torch.amp.autocast('cuda'):
                    pred_pos = m(h_b, lv_b, lv_raw_b, init_b, hist_mean_t, hist_std_t,
                                 fut_true_pos=tp_b, teacher_forcing_ratio=tf_ratio)
                    pos_loss = torch.nn.functional.smooth_l1_loss(pred_pos, tp_b, beta=1.0)
                    pred_vel = torch.diff(pred_pos, dim=1) / dt
                    true_vel = torch.diff(tp_b, dim=1) / dt
                    vel_loss = torch.nn.functional.smooth_l1_loss(pred_vel, true_vel, beta=1.0)
                    pos_lv_b = lv_raw_b[:, :, 2]
                    time_weights = torch.linspace(1.0, 5.0, HORIZON, device=h_b.device)
                    headway_err = (pos_lv_b - pred_pos) - (pos_lv_b - tp_b)
                    headway_loss = torch.mean(time_weights * torch.nn.functional.smooth_l1_loss(
                        headway_err, torch.zeros_like(headway_err), beta=1.0, reduction='none'))
                    pred_acc2 = torch.diff(pred_vel, dim=1) / dt
                    true_acc2 = torch.diff(true_vel, dim=1) / dt
                    acc_loss = torch.nn.functional.smooth_l1_loss(pred_acc2, true_acc2, beta=1.0)
                    pred_jerk = torch.diff(pred_acc2, dim=1) / dt
                    jerk_loss = torch.mean(pred_jerk ** 2)
                    loss = 0.05 * pos_loss + 0.20 * vel_loss + 0.60 * headway_loss + 0.15 * acc_loss + JERK_WEIGHT * jerk_loss
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
            m.eval()
            with torch.no_grad():
                val_pred_pos = m(val_hist, val_fut_lv, val_fut_lv_raw, val_init, hist_mean_t, hist_std_t,
                                 teacher_forcing_ratio=0.0).cpu().numpy()
            pred_rows = []
            for i, seg_id in enumerate(val_data['seg_ids']):
                for t in range(HORIZON):
                    pred_rows.append({'Segment_ID': seg_id, 'Time_Index': round(10.0 + t*dt, 2), 'Pos_FAV': val_pred_pos[i, t]})
            df_pred = pd.DataFrame(pred_rows)
            score = evaluate_predictions(df_true=df_val_horizon, df_pred=df_pred, dt=dt)
            if score > best_m_score:
                best_m_score = score
                torch.save(m.state_dict(), f"ensemble_kaggle_{member_idx}.pt")
                epochs_since_best = 0
            else:
                epochs_since_best += 1
            if epochs_since_best >= 10 and tf_ratio == 0.0:
                break
        print(f"Member {member_idx+1} best: {best_m_score:.4f}")
    print("Training complete.")
else:
    print("Skipping training (TRAIN=False). Using existing checkpoints.")

# ============================================================================
# 6. EVALUATION & SUBMISSION
# ============================================================================
# Load validation data (needed for stats and evaluation)
full_df = pd.read_csv(TRAIN_PATH)
unique_segments = full_df['Segment_ID'].unique()
np.random.seed(42)
val_segments = set(np.random.choice(unique_segments, size=int(len(unique_segments) * 0.03), replace=False))
val_segments = list(val_segments)

# Build validation tensors
val_data = build_segment_tensors(val_segments, full_df)

# We need the normalisation stats from the full training set (or use the ones from training)
# To be safe, compute from the entire training set (same as before)
all_train_segments = [s for s in unique_segments if s not in val_segments]
train_data_stats = build_segment_tensors(all_train_segments, full_df)
hist_mean = train_data_stats['hist'].reshape(-1, 11).mean(axis=0)
hist_std  = train_data_stats['hist'].reshape(-1, 11).std(axis=0) + 1e-6
lv_mean   = train_data_stats['fut_lv'].reshape(-1, 3).mean(axis=0)
lv_std    = train_data_stats['fut_lv'].reshape(-1, 3).std(axis=0) + 1e-6

hist_mean_t = torch.tensor(hist_mean, device=device)
hist_std_t  = torch.tensor(hist_std, device=device)

val_hist = torch.tensor((val_data['hist'] - hist_mean) / hist_std, device=device)
val_fut_lv = torch.tensor((val_data['fut_lv'] - lv_mean) / lv_std, device=device)
val_fut_lv_raw = torch.tensor(np.stack([
    val_data['fut_lv'][:, :, 0], val_data['fut_lv'][:, :, 1],
    val_data['fut_lv'][:, :, 2] + val_data['init_state'][:, 0:1]
], axis=-1), device=device, dtype=torch.float32)
val_init = torch.tensor(val_data['init_state'], device=device)

df_val_horizon = full_df[full_df['Segment_ID'].isin(val_data['seg_ids']) & (full_df['Time_Index'] >= 10.00)].copy()

# Load ensemble models and evaluate
N_ENSEMBLE = 5
all_accels = []
for i in range(N_ENSEMBLE):
    m = EnhancedLSTM().to(device)
    m.load_state_dict(torch.load(f"ensemble_kaggle_{i}.pt", map_location=device))
    m.eval()
    with torch.no_grad():
        accels = forward_with_accel(m, val_hist, val_fut_lv, val_fut_lv_raw, val_init, hist_mean_t, hist_std_t)
    all_accels.append(accels.cpu().numpy())
    del m
    torch.cuda.empty_cache()
avg_accels = np.mean(all_accels, axis=0)

# Optional: evaluate with best cutoff (from validation)
init_np = val_data['init_state']
pred_rows = []
for i, seg_id in enumerate(val_data['seg_ids']):
    filtered = apply_lowpass(avg_accels[i], SMOOTHING_CUTOFF)
    positions = integrate(init_np[i, 0], init_np[i, 1], filtered)
    for t, p in enumerate(positions):
        pred_rows.append({'Segment_ID': seg_id, 'Time_Index': round(10.0 + t*dt, 2), 'Pos_FAV': p})
df_pred = pd.DataFrame(pred_rows)
print(f"\n=== Ensemble validation with {SMOOTHING_CUTOFF} Hz smoothing ===")
evaluate_predictions(df_true=df_val_horizon, df_pred=df_pred, dt=dt)

# ---- Generate test submission ----
print("\nGenerating test submission...")
test_df = pd.read_csv(TEST_PATH)
test_data = build_test_tensors(test_df)

test_hist = torch.tensor((test_data['hist'] - hist_mean) / hist_std, device=device)
test_fut_lv = torch.tensor((test_data['fut_lv'] - lv_mean) / lv_std, device=device)
test_fut_lv_raw = torch.tensor(test_data['fut_lv_raw'], device=device, dtype=torch.float32)
test_init = torch.tensor(test_data['init_state'], device=device)

N_test = test_hist.shape[0]
BATCH_SIZE = 128
all_test_accels = []
for i in range(N_ENSEMBLE):
    m = EnhancedLSTM().to(device)
    m.load_state_dict(torch.load(f"ensemble_kaggle_{i}.pt", map_location=device))
    m.eval()
    acc_batches = []
    with torch.no_grad():
        for start in range(0, N_test, BATCH_SIZE):
            end = min(start + BATCH_SIZE, N_test)
            hb = test_hist[start:end]
            lb = test_fut_lv[start:end]
            lrb = test_fut_lv_raw[start:end]
            ib = test_init[start:end]
            a = forward_with_accel(m, hb, lb, lrb, ib, hist_mean_t, hist_std_t)
            acc_batches.append(a.cpu().numpy())
    all_test_accels.append(np.concatenate(acc_batches, axis=0))
    del m
    torch.cuda.empty_cache()

avg_test_accels = np.mean(all_test_accels, axis=0)
pred_rows = []
for i, seg_id in enumerate(test_data['seg_ids']):
    filtered = apply_lowpass(avg_test_accels[i], SMOOTHING_CUTOFF)
    positions = integrate(test_data['init_state'][i, 0], test_data['init_state'][i, 1], filtered)
    for t, p in enumerate(positions):
        pred_rows.append({'Segment_ID': seg_id, 'Time_Index': round(10.0 + t*dt, 2), 'Pos_FAV': p})

df_sub = pd.DataFrame(pred_rows)
df_sub.to_csv('submission.csv', index=False)
print("Submission saved as 'submission.csv'")
print(df_sub.head())

assert set(df_sub.columns) == {'Segment_ID','Time_Index','Pos_FAV'}
counts = df_sub.groupby('Segment_ID').size()
assert counts.eq(200).all()
print("✅ Submission valid. Upload now!")