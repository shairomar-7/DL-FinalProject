"""Milestone pipeline: audio-only baseline under noise + PianoVAM-style rule-based video filter.

Data layout (produced by 01_preprocess.ipynb):
  ROOT/metadata.json, ROOT/Audio16k/<id>.wav, ROOT/Landmarks/<id>.npz, ROOT/TSV/<id>.tsv
"""
import os, json
import numpy as np
import soundfile as sf
import mir_eval

SR = 16000          # audio sample rate (Kong et al. model expects 16 kHz)
FPS_V = 60          # PianoVAM video / landmark frame rate
VID_W, VID_H = 1920, 1080   # PianoVAM video resolution (keyboard corners are in these pixels)
N_WHITE = 52        # white keys on an 88-key piano (A0..C8)
FINGERTIPS = [4, 8, 12, 16, 20]  # MediaPipe Hands fingertip landmark indices


# ---------------------------------------------------------------- data
def load_meta(root):
    meta = json.load(open(f'{root}/metadata.json'))
    return {v['record_time']: v for v in meta.values()}


def split_ids(meta, split):
    return sorted(i for i, v in meta.items() if v['split'] == split)


def load_audio(root, rid):
    y, sr = sf.read(f'{root}/Audio16k/{rid}.wav', dtype='float32')
    assert sr == SR, sr
    return y


def load_ref_notes(root, rid):
    """Ground-truth notes from PianoVAM TSV: columns onset, key_offset, frame_offset, note, velocity."""
    n = np.loadtxt(f'{root}/TSV/{rid}.tsv', skiprows=1, ndmin=2)
    onset, off = n[:, 0], n[:, 2]
    off = np.maximum(off, onset + 1e-2)          # mir_eval needs offset > onset
    return np.stack([onset, off], 1), n[:, 3].astype(int)


def crop_notes(intervals, pitches, max_sec):
    keep = intervals[:, 0] < max_sec
    return intervals[keep], pitches[keep]


# ---------------------------------------------------------------- audio degradation
def add_white_noise(y, snr_db, seed=0):
    """Additive white Gaussian noise at a given SNR (dB) w.r.t. the recording's mean power."""
    if snr_db is None:
        return y
    rng = np.random.default_rng(seed)
    p_sig = np.mean(y.astype(np.float64) ** 2)
    p_noise = p_sig / (10 ** (snr_db / 10))
    return (y + rng.normal(0, np.sqrt(p_noise), size=y.shape)).astype(np.float32)


# ---------------------------------------------------------------- audio-only transcription
def notes_from_events(events):
    if len(events) == 0:
        return np.zeros((0, 2)), np.zeros(0, int)
    iv = np.array([[e['onset_time'], max(e['offset_time'], e['onset_time'] + 1e-2)] for e in events])
    p = np.array([int(e['midi_note']) for e in events])
    ok = np.isfinite(iv).all(1)            # the regression post-processor can occasionally emit NaN times
    return iv[ok], p[ok]


def transcribe(transcriptor, y):
    out = transcriptor.transcribe(y, None)
    return notes_from_events(out['est_note_events'])


# ---------------------------------------------------------------- evaluation
def note_counts(ref_iv, ref_p, est_iv, est_p, onset_tol=0.05):
    """Onset-only note matching (mir_eval, 50 ms). Returns (n_matched, n_ref, n_est)."""
    if len(est_p) == 0 or len(ref_p) == 0:
        return 0, len(ref_p), len(est_p)
    m = mir_eval.transcription.match_notes(
        ref_iv, mir_eval.util.midi_to_hz(ref_p), est_iv, mir_eval.util.midi_to_hz(est_p),
        onset_tolerance=onset_tol, offset_ratio=None)
    return len(m), len(ref_p), len(est_p)


def prf(tp, n_ref, n_est):
    p = tp / n_est if n_est else 0.0
    r = tp / n_ref if n_ref else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


# ---------------------------------------------------------------- hands -> keyboard coordinates
def homography(corners):
    """4 pixel corners (LT, RT, RB, LB) -> unit square (u: left->right along keyboard, v: top->bottom)."""
    dst = [(0, 0), (1, 0), (1, 1), (0, 1)]
    A = []
    for (x, y), (u, v) in zip(corners, dst):
        A += [[x, y, 1, 0, 0, 0, -u * x, -u * y, -u], [0, 0, 0, x, y, 1, -v * x, -v * y, -v]]
    _, _, Vt = np.linalg.svd(np.array(A, float))
    return Vt[-1].reshape(3, 3)


def keyboard_corners(m):
    return [tuple(map(float, m[k].split(','))) for k in ['Point_LT', 'Point_RT', 'Point_RB', 'Point_LB']]


def fingertips_on_keyboard(root, rid, m):
    """Returns (T, 10) array: fingertip position along the keyboard in white-key units (0..52); NaN if hand missing."""
    X = np.load(f'{root}/Landmarks/{rid}.npz')['landmarks'].astype(np.float32)   # (T, 2, 21, 3)
    tips = X[:, :, FINGERTIPS, :2].reshape(-1, 2) * [VID_W, VID_H]            # normalized -> pixels
    H = homography(keyboard_corners(m))
    p = np.c_[tips, np.ones(len(tips))] @ H.T
    u = (p[:, 0] / p[:, 2]).reshape(X.shape[0], 10)
    return u * N_WHITE


_WHITE_OFF = [0, None, 1, 2, None, 3, None, 4, 5, None, 6, None]   # A A# B C C# D D# E F F# G G#


def key_position(pitch):
    """Horizontal position of a MIDI pitch (21..108) in white-key units; black keys sit between whites."""
    octv, r = divmod(int(pitch) - 21, 12)
    if _WHITE_OFF[r] is not None:
        return octv * 7 + _WHITE_OFF[r] + 0.5
    return octv * 7 + _WHITE_OFF[r - 1] + 1.0


def nearest_tip_distance(tips_u, onsets, pitches, win=2):
    """Distance (white keys) from each note's key to the nearest fingertip within +-win frames of its onset.
    NaN when no hand is detected in that window."""
    T = tips_u.shape[0]
    kp = np.array([key_position(p) for p in pitches])
    d = np.full(len(onsets), np.nan)
    for j, (t, k) in enumerate(zip(onsets, kp)):
        f = int(round(t * FPS_V))
        w = tips_u[max(f - win, 0):min(f + win + 1, T)]
        if w.size and not np.all(np.isnan(w)):
            d[j] = np.nanmin(np.abs(w - k))
    return d


def rule_filter(est_iv, est_p, tips_u, tol=2.0, win=2):
    """PianoVAM-style post-processing: delete a predicted note if no fingertip is within `tol` white keys
    of its key at onset. Notes with no detected hand are kept (the rule cannot judge them)."""
    if len(est_p) == 0:
        return est_iv, est_p
    d = nearest_tip_distance(tips_u, est_iv[:, 0], est_p, win)
    keep = np.isnan(d) | (d <= tol)
    return est_iv[keep], est_p[keep]
