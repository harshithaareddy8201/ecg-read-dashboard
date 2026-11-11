"""ECG report explainer: analysis pipeline.

Detects heartbeats, measures rate and rhythm, flags beats that come early or look
unusual, and checks the results against cardiologist labels where they exist.

Data (public, MIT-BIH Arrhythmia Database via PhysioNet):
  - Record 100: 30 min, 2-lead, 360 Hz, every beat labelled by cardiologists.
  - Record 208: 5-min excerpt shipped with SciPy (scipy.datasets.electrocardiogram); no labels.

Run:  pip install wfdb neurokit2 numpy pandas pooch
      python analyze.py   -> writes out/results.json for the dashboard
"""
import json, base64
from pathlib import Path
import numpy as np, wfdb, neurokit2 as nk

HERE = Path(__file__).parent
FS = 360
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)


def detect(sig):
    clean = nk.ecg_clean(sig, sampling_rate=FS, method="neurokit")
    _, info = nk.ecg_peaks(clean, sampling_rate=FS, method="neurokit")
    return clean, np.asarray(info["ECG_R_Peaks"])


def classify(clean, peaks):
    """Simple, explainable rules (no black box):
    early   = RR interval < 85% of the median of the previous 8 RR intervals
    unusual = shape correlates < 0.80 with this patient's median beat
    label   : normal | early (normal shape) | unusual shape
    """
    w0, w1 = int(0.25 * FS), int(0.40 * FS)
    ok = (peaks > w0) & (peaks < len(clean) - w1)
    peaks = peaks[ok]
    beats = np.stack([clean[p - w0:p + w1] for p in peaks])
    template = np.median(beats, axis=0)
    corr = np.array([np.corrcoef(b, template)[0, 1] for b in beats])
    rr = np.diff(peaks) / FS
    early = np.zeros(len(peaks), bool)
    for i in range(1, len(peaks)):
        ref = np.median(rr[max(0, i - 9):i - 1]) if i > 1 else rr[0]
        early[i] = rr[i - 1] < 0.85 * ref
    unusual = corr < 0.80
    label = np.where(unusual, "unusual", np.where(early, "early", "normal"))
    ratio = np.ones(len(peaks))
    for i in range(1, len(peaks)):
        ref = np.median(rr[max(0, i - 9):i - 1]) if i > 1 else rr[0]
        ratio[i] = rr[i - 1] / ref
    classify.ratio = ratio
    return peaks, label, corr, template


def summarise(peaks, label):
    rr = np.diff(peaks) / FS
    hr = 60 / rr
    dur_min = (peaks[-1] - peaks[0]) / FS / 60
    n = len(label)
    early, unusual = int((label == "early").sum()), int((label == "unusual").sum())
    normal_rr = rr[(label[1:] == "normal") & (label[:-1] == "normal")]
    # per-minute heart rate for the trend chart
    minute = (peaks[1:] / FS // 60).astype(int)
    per_min = [{"minute": int(m), "bpm": round(float(np.median(hr[minute == m])), 1)} for m in np.unique(minute)]
    burden = (early + unusual) / n * 100
    avg = float(np.median(hr))
    if avg < 50 or avg > 110 or unusual / n > 0.10:
        verdict, advice = "red", "See a doctor soon. The rhythm shows frequent irregular beats or a heart rate outside the usual range."
    elif burden > 1:
        verdict, advice = "amber", "Mention this at your next check-up. Some extra beats were seen; they are often harmless but worth a doctor's look."
    else:
        verdict, advice = "green", "Looks regular. Heart rate is in the usual range and extra beats are rare."
    return {
        "duration_min": round(dur_min, 1), "beats": n,
        "hr_avg": round(avg, 0), "hr_min": round(float(np.percentile(hr, 2)), 0),
        "hr_max": round(float(np.percentile(hr, 98)), 0),
        "sdnn_ms": round(float(np.std(normal_rr) * 1000), 0),
        "early": early, "unusual": unusual, "burden_pct": round(burden, 2),
        "verdict": verdict, "advice": advice, "hr_per_min": per_min,
    }


def evaluate(peaks, label, ann):
    """Match detected beats to cardiologist labels within 100 ms."""
    keep = np.isin(ann.symbol, list("NLRAaJSVEFje/fQ"))
    ref_idx, ref_sym = ann.sample[keep], np.array(ann.symbol)[keep]
    tol = int(0.10 * FS)
    j = np.searchsorted(ref_idx, peaks)
    matched, rows = 0, []
    used = set()
    for p, lab in zip(peaks, label):
        cand = [k for k in (j[peaks.tolist().index(p)] - 1, j[peaks.tolist().index(p)]) if 0 <= k < len(ref_idx)]
        best = min(cand, key=lambda k: abs(ref_idx[k] - p)) if cand else None
        if best is not None and abs(ref_idx[best] - p) <= tol and best not in used:
            used.add(best); matched += 1
            rows.append((ref_sym[best], lab))
    sens = matched / len(ref_idx); ppv = matched / len(peaks)
    groups = {"N": "Normal", "A": "Atrial premature", "V": "Ventricular premature"}
    conf = {}
    for s, lab in rows:
        g = groups.get(s, "Other")
        conf.setdefault(g, {"normal": 0, "early": 0, "unusual": 0})[lab] += 1
    return {"reference_beats": int(len(ref_idx)), "detected_beats": int(len(peaks)),
            "sensitivity_pct": round(sens * 100, 2), "ppv_pct": round(ppv * 100, 2), "confusion": conf}


def pack(sig):
    q = np.clip(np.round(sig * 1000), -32768, 32767).astype("<i2")  # millivolts * 1000
    return base64.b64encode(q.tobytes()).decode()


results = {}
# ---- Record 100: labelled ----
rec = wfdb.rdrecord(str(HERE / "data" / "100")); ann = wfdb.rdann(str(HERE / "data" / "100"), "atr")
sig = rec.p_signal[:, 0]
clean, peaks = detect(sig)
peaks, label, corr, tmpl = classify(clean, peaks)
results["rec100"] = {"name": "Record 100", "note": "30-minute recording, cardiologist-labelled",
                     "summary": summarise(peaks, label), "evaluation": evaluate(peaks, label, ann),
                     "signal": pack(clean), "peaks": peaks.tolist(), "labels": label.tolist(),
                     "template": np.round(tmpl, 3).tolist(), "corr": np.round(corr, 4).tolist(), "ratio": np.round(classify.ratio, 4).tolist()}
# ---- Record 208 excerpt: unlabelled ----
sig = np.load(HERE / "data" / "rec208_excerpt.npy")
clean, peaks = detect(sig)
peaks, label, corr, tmpl = classify(clean, peaks)
results["rec208"] = {"name": "Record 208", "note": "5-minute excerpt, no beat labels available here",
                     "summary": summarise(peaks, label), "evaluation": None,
                     "signal": pack(clean), "peaks": peaks.tolist(), "labels": label.tolist(),
                     "template": np.round(tmpl, 3).tolist(), "corr": np.round(corr, 4).tolist(), "ratio": np.round(classify.ratio, 4).tolist()}

(OUT / "results.json").write_text(json.dumps(results))
for k, v in results.items():
    print(k, {kk: vv for kk, vv in v["summary"].items() if kk != "hr_per_min"})
    print("  eval:", v["evaluation"])
