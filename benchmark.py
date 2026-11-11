"""Full MIT-BIH benchmark: simple rules vs a trained classifier, tested on unseen patients.

Data: MIT-BIH Arrhythmia Database (48 records), unzipped into ./mitdb_raw/mit-bih-arrhythmia-database-1.0.0/
Split: de Chazal et al. (2004) inter-patient split. DS1 = training patients, DS2 = test patients.
Paced records (102, 104, 107, 217) are excluded, as is standard.
Beat classes (AAMI): N normal, S supraventricular premature (e.g. atrial), V ventricular premature.
Run: pip install wfdb neurokit2 scikit-learn pandas numpy ; python benchmark.py
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, wfdb, neurokit2 as nk
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
HERE = Path(__file__).parent
DB = HERE / "mitdb_raw" / "mit-bih-arrhythmia-database-1.0.0"
OUT = HERE / "out" / "csv"; OUT.mkdir(parents=True, exist_ok=True)
FS = 360
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]
AAMI = {**{s: "N" for s in "NLRej"}, **{s: "S" for s in "AaJS"}, **{s: "V" for s in "VE"}, "F": "F", **{s: "Q" for s in "/fQ"}}
W0, W1 = int(0.25 * FS), int(0.40 * FS)


def features(rec_id):
    rec = wfdb.rdrecord(str(DB / str(rec_id))); ann = wfdb.rdann(str(DB / str(rec_id)), "atr")
    sig = rec.p_signal[:, 0]
    clean = nk.ecg_clean(sig, sampling_rate=FS, method="neurokit")
    _, info = nk.ecg_peaks(clean, sampling_rate=FS, method="neurokit")
    peaks = np.asarray(info["ECG_R_Peaks"])
    peaks = peaks[(peaks > W0) & (peaks < len(clean) - W1)]
    beats = np.stack([clean[p - W0:p + W1] for p in peaks])
    template = np.median(beats, axis=0)
    corr = np.array([np.corrcoef(b, template)[0, 1] for b in beats])
    rr = np.diff(peaks) / FS
    rr_prev = np.r_[rr[0], rr]; rr_next = np.r_[rr, rr[-1]]
    local = pd.Series(rr_prev).rolling(8, min_periods=1).median().shift(1).bfill().to_numpy()
    ratio = rr_prev / local
    # beat shape: 24 points from the window, scaled by this record's median R height
    scale = np.median(np.abs(beats[:, W0])) or 1
    shape = beats[:, ::int(len(template) / 24)][:, :24] / scale
    width = (np.abs(np.diff(beats[:, W0 - 30:W0 + 30], axis=1)) > 0.15 * np.abs(np.diff(beats, axis=1)).max(axis=1, keepdims=True)).sum(axis=1)
    # match detected beats to labels (100 ms)
    keep = np.isin(ann.symbol, list(AAMI))
    ref_idx = ann.sample[keep]; ref_cls = np.array([AAMI[s] for s in np.array(ann.symbol)[keep]])
    j = np.clip(np.searchsorted(ref_idx, peaks), 1, len(ref_idx) - 1)
    near = np.where(np.abs(ref_idx[j - 1] - peaks) < np.abs(ref_idx[j] - peaks), j - 1, j)
    ok = np.abs(ref_idx[near] - peaks) <= 0.1 * FS
    truth = np.where(ok, ref_cls[near], "none")
    found = np.zeros(len(ref_idx), bool); found[near[ok]] = True
    df = pd.DataFrame({"record": rec_id, "truth": truth, "rr_prev": rr_prev, "rr_next": rr_next, "ratio": ratio,
                       "next_ratio": rr_next / local, "corr": corr, "width": width, "amp": beats[:, W0] / scale})
    df = pd.concat([df, pd.DataFrame(shape, columns=[f"s{i}" for i in range(shape.shape[1])])], axis=1)
    det = {"record": rec_id, "labelled_beats": int(len(ref_idx)), "detected_beats": int(len(peaks)),
           "beats_found": int(found.sum()), "false_detections": int((~ok).sum()),
           "missed_by_class": {c: int(((ref_cls == c) & ~found).sum()) for c in "NSV"},
           "total_by_class": {c: int((ref_cls == c).sum()) for c in "NSV"}}
    return df, det


import pickle
CACHE = HERE / "out" / "features.pkl"
if CACHE.exists():
    data, dets = pickle.loads(CACHE.read_bytes())
else:
    rows, dets = [], []
    for r in DS1 + DS2:
        df, det = features(r); rows.append(df); dets.append(det); print("done", r, det["labelled_beats"], det["beats_found"])
    data = pd.concat(rows, ignore_index=True)
    CACHE.write_bytes(pickle.dumps((data, dets)))
# gaps relative to this patient's typical gap
med = data.groupby("record").rr_prev.transform("median")
data["rr_rec"] = data.rr_prev / med; data["rrn_rec"] = data.rr_next / med; data["rr_diff"] = data.rr_next - data.rr_prev
data["rule"] = np.where(data["corr"] < 0.80, "V", np.where(data["ratio"] < 0.85, "S", "N"))
FEATS = ["rr_prev", "rr_next", "ratio", "next_ratio", "corr", "width", "amp", "rr_rec", "rrn_rec", "rr_diff"] + [f"s{i}" for i in range(24)]

train = data[data.record.isin(DS1) & data.truth.isin(["N", "S", "V"])]
test = data[data.record.isin(DS2)].copy()
from sklearn.model_selection import GroupKFold
RF = dict(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=7)
train = train.reset_index(drop=True)
# choose the atrial-beat cut-off by cross-validation across TRAINING patients only (test patients untouched)
proba = np.zeros((len(train), 3))
for a, b in GroupKFold(5).split(train, train.truth, train.record):
    m = RandomForestClassifier(**RF).fit(train.loc[a, FEATS], train.truth[a]); proba[b] = m.predict_proba(train.loc[b, FEATS])
ci = {c: i for i, c in enumerate(m.classes_)}
best = None
for th in [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1]:
    pred = np.where(proba[:, ci["V"]] > 0.5, "V", np.where(proba[:, ci["S"]] > th, "S", "N"))
    S = train.truth == "S"; sens = (pred[S] == "S").mean(); prec = (train.truth[pred == "S"] == "S").mean() if (pred == "S").any() else 0
    f1 = 2 * sens * prec / max(sens + prec, 1e-9); print(f"atrial cut-off {th}: CV sensitivity {sens:.2f}, precision {prec:.2f}")
    if best is None or f1 > best[1]: best = (th, f1)
S_CUT = best[0]; print("chosen atrial cut-off", S_CUT)
model = RandomForestClassifier(**RF).fit(train[FEATS], train.truth)
p = model.predict_proba(test[FEATS]); ci = {c: i for i, c in enumerate(model.classes_)}
test["model"] = np.where(p[:, ci["V"]] > 0.5, "V", np.where(p[:, ci["S"]] > S_CUT, "S", "N"))

def scores(pred_col, df):
    d = df[df.truth.isin(["N", "S", "V"])]
    out = []
    for c in "NSV":
        tp = int(((d.truth == c) & (d[pred_col] == c)).sum()); fn = int(((d.truth == c) & (d[pred_col] != c)).sum())
        fp = int(((d.truth != c) & (d[pred_col] == c)).sum())
        out.append({"beat_type": c, "true_beats": tp + fn, "caught": tp, "sensitivity_pct": round(100 * tp / max(tp + fn, 1), 1),
                    "precision_pct": round(100 * tp / max(tp + fp, 1), 1), "false_alarms": fp})
    return out

name = {"N": "Normal", "S": "Atrial / supraventricular premature", "V": "Ventricular premature"}
cmp_rows = []
for method, col in [("Simple rules", "rule"), ("Trained model", "model")]:
    for s in scores(col, test):
        cmp_rows.append({"method": method, "beat_type": name[s["beat_type"]], **{k: v for k, v in s.items() if k != "beat_type"}})
cmp = pd.DataFrame(cmp_rows)
cmp.to_csv(OUT / "method_comparison.csv", index=False)
print(cmp.to_string(index=False))

# per-record table for all 44 records (rules + detection; model on test records)
per = []
for det in dets:
    r = det["record"]; d = data[data.record == r]; lab = d[d.truth.isin(["N", "S", "V"])]
    row = {"record": str(r), "split": "train" if r in DS1 else "test", "labelled_beats": det["labelled_beats"],
           "detection_pct": round(100 * det["beats_found"] / det["labelled_beats"], 2),
           "false_detections": det["false_detections"],
           "ventricular_beats": det["total_by_class"]["V"], "atrial_beats": det["total_by_class"]["S"],
           "rules_v_caught": int(((lab.truth == "V") & (lab.rule == "V")).sum()),
           "rules_s_caught": int(((lab.truth == "S") & (lab.rule == "S")).sum()),
           "rules_false_alarms": int(((lab.truth == "N") & (lab.rule != "N")).sum())}
    if r in DS2:
        t = test[test.record == r]; t = t[t.truth.isin(["N", "S", "V"])]
        row.update({"model_v_caught": int(((t.truth == "V") & (t.model == "V")).sum()),
                    "model_s_caught": int(((t.truth == "S") & (t.model == "S")).sum()),
                    "model_false_alarms": int(((t.truth == "N") & (t.model != "N")).sum())})
    per.append(row)
per = pd.DataFrame(per).sort_values("record")
for c in ["model_v_caught", "model_s_caught", "model_false_alarms"]: per[c] = per[c].astype("Int64")
per.to_csv(OUT / "all_records.csv", index=False)

tot = {"records": len(dets), "labelled_beats": sum(d["labelled_beats"] for d in dets),
       "beats_found": sum(d["beats_found"] for d in dets), "false_detections": sum(d["false_detections"] for d in dets)}
tot["detection_pct"] = round(100 * tot["beats_found"] / tot["labelled_beats"], 2)
tot["precision_pct"] = round(100 * tot["beats_found"] / (tot["beats_found"] + tot["false_detections"]), 2)
(OUT / "benchmark_totals.csv").write_text("metric,value\n" + "".join(f"{k},{v}\n" for k, v in tot.items()))
print(tot)
imp = pd.Series(model.feature_importances_, FEATS).sort_values(ascending=False).head(8)
print(imp)
