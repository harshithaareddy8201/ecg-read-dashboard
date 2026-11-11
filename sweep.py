"""Threshold trade-off: how flags, misses and false alarms change with the two rule thresholds."""
import json, numpy as np, pandas as pd, wfdb
from pathlib import Path
HERE = Path(__file__).parent; FS = 360
R = json.loads((HERE / "out" / "results.json").read_text())
ann = wfdb.rdann(str(HERE / "data" / "100"), "atr")
keep = np.isin(ann.symbol, list("NLRAaJSVEFje/fQ"))
ref_idx, ref_sym = ann.sample[keep], np.array(ann.symbol)[keep]
EARLY = [0.75, 0.80, 0.85, 0.90, 0.95]
SHAPE = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
rows = []
for key, rid in [("rec100", "100"), ("rec208", "208")]:
    v = R[key]; peaks = np.array(v["peaks"]); corr = np.array(v["corr"]); ratio = np.array(v["ratio"])
    truth = None
    if rid == "100":   # match detected beats to cardiologist labels
        truth = np.array(["?"] * len(peaks), dtype=object)
        for i, p in enumerate(peaks):
            j = np.abs(ref_idx - p).argmin()
            if abs(ref_idx[j] - p) <= 0.1 * FS: truth[i] = ref_sym[j]
        abnormal_total = int(np.isin(ref_sym, ["A", "V"]).sum())   # includes beats the detector never found
    for e in EARLY:
        for s in SHAPE:
            unusual = corr < s; early = (ratio < e) & ~unusual; flagged = unusual | early
            r = {"record": rid, "early_threshold": e, "shape_threshold": s, "flagged_early": int(early.sum()),
                 "flagged_unusual": int(unusual.sum()), "flagged_pct": round(flagged.mean() * 100, 2)}
            if truth is not None:
                abn = np.isin(truth, ["A", "V"]); nor = truth == "N"
                r.update({"abnormal_beats": abnormal_total, "abnormal_caught": int((flagged & abn).sum()),
                          "abnormal_missed": abnormal_total - int((flagged & abn).sum()),
                          "false_alarms": int((flagged & nor).sum()), "normal_beats": int(nor.sum())})
            rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(HERE / "out" / "csv" / "threshold_sweep.csv", index=False)
print(df[df.record == "100"].pivot(index="early_threshold", columns="shape_threshold", values="false_alarms"))
print(df[df.record == "100"].pivot(index="early_threshold", columns="shape_threshold", values="abnormal_caught"))
print(df[df.record == "208"].pivot(index="early_threshold", columns="shape_threshold", values="flagged_pct"))
