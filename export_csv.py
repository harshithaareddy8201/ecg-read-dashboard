"""Turn analysis results into the CSV files the dashboard reads."""
import json, base64, numpy as np, pandas as pd, wfdb
from pathlib import Path
HERE = Path(__file__).parent; OUT = HERE / "out"; FS = 360
R = json.loads((OUT / "results.json").read_text())
names = {"rec100": "100", "rec208": "208"}

rec_rows, hr_rows, beat_rows, strip_rows, strip_index = [], [], [], [], []
for key, v in R.items():
    rid = names[key]; s = v["summary"]
    rec_rows.append({"record": rid, "description": v["note"], **{k: s[k] for k in
        ["duration_min", "beats", "hr_avg", "hr_min", "hr_max", "sdnn_ms", "early", "unusual", "burden_pct", "verdict", "advice"]}})
    hr_rows += [{"record": rid, "minute": m["minute"] + 1, "bpm": m["bpm"]} for m in s["hr_per_min"]]
    peaks = np.array(v["peaks"]); lab = np.array(v["labels"])
    rr = np.r_[np.nan, np.diff(peaks) / FS * 1000]
    rr_next = np.r_[rr[1:], np.nan]
    for i, (p, l) in enumerate(zip(peaks, lab)):
        beat_rows.append({"record": rid, "beat": i + 1, "time_s": round(p / FS, 3), "label": l,
                          "rr_ms": None if np.isnan(rr[i]) else round(rr[i]), "rr_next_ms": None if np.isnan(rr_next[i]) else round(rr_next[i])})
    sig = np.frombuffer(base64.b64decode(v["signal"]), dtype="<i2") / 1000
    # strips: one typical 10 s window, then up to 8 windows centred on flagged beats, spread across the recording
    flagged = peaks[lab != "normal"]
    picks = flagged[np.linspace(0, len(flagged) - 1, min(8, len(flagged))).round().astype(int)] if len(flagged) else []
    windows = [("first", 0, None)] + [("flagged", max(0, int(p - 5 * FS)), p) for p in picks if p > 10 * FS]
    for n, (kind, start, beat) in enumerate(windows, 1):
        end = min(len(sig), start + 10 * FS)
        sid = f"{rid}-{n}"
        t0 = start / FS
        mm, ss = divmod(int(beat / FS) if beat is not None else 0, 60)
        lab_txt = "First 10 seconds" if kind == "first" else f"Flagged beat at {mm}:{ss:02d}"
        strip_index.append({"record": rid, "strip": sid, "order": n, "kind": kind, "start_s": round(t0, 2), "label": lab_txt})
        for i in range(start, end, 2):  # 180 Hz is plenty for display
            strip_rows.append({"strip": sid, "t": round(i / FS - t0, 4), "mv": round(float(sig[i]), 3)})

# accuracy on record 100 (cardiologist labels)
ann = wfdb.rdann(str(HERE / "data" / "100"), "atr")
keep = np.isin(ann.symbol, list("NLRAaJSVEFje/fQ"))
ref_idx, ref_sym = ann.sample[keep], np.array(ann.symbol)[keep]
peaks = np.array(R["rec100"]["peaks"]); lab = np.array(R["rec100"]["labels"])
groups = {"N": "Normal", "A": "Atrial premature", "V": "Ventricular premature"}
conf = {g: {"normal": 0, "early": 0, "unusual": 0, "missed": 0} for g in groups.values()}
for idx, s in zip(ref_idx, ref_sym):
    d = np.abs(peaks - idx); j = d.argmin()
    g = groups.get(s, "Normal")
    if d[j] <= 0.1 * FS: conf[g][lab[j]] += 1
    else: conf[g]["missed"] += 1
eval_rows = [{"cardiologist_label": g, "beats": sum(c.values()), "rated_normal": c["normal"], "flagged_early": c["early"],
              "flagged_unusual": c["unusual"], "missed": c["missed"]} for g, c in conf.items()]
ev = R["rec100"]["evaluation"]
metrics = [{"metric": "Beats labelled by cardiologists", "value": ev["reference_beats"]},
           {"metric": "Beats detected", "value": ev["detected_beats"]},
           {"metric": "Detection sensitivity (%)", "value": ev["sensitivity_pct"]},
           {"metric": "Detection precision (%)", "value": ev["ppv_pct"]}]

D = OUT / "csv"; D.mkdir(exist_ok=True)
for name, rows in {"recordings": rec_rows, "hr_trend": hr_rows, "beats": beat_rows, "strip_index": strip_index,
                   "strips": strip_rows, "accuracy_by_class": eval_rows, "detection": metrics}.items():
    pd.DataFrame(rows).to_csv(D / f"{name}.csv", index=False)
    print(f"{name:18s} {len(rows):>7} rows  {(D / f'{name}.csv').stat().st_size/1024:8.0f} KB")
print(pd.DataFrame(eval_rows).to_string(index=False))
print(pd.DataFrame(strip_index).to_string(index=False))
