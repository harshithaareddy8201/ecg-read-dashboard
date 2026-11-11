# ECG Report Explainer

**Turning a raw ECG signal into an answer a patient can act on.**

Most people who get an ECG see one word printed above the strip: *normal* or *abnormal*. That word doesn't tell them what is happening or what to do next, so they either panic or ignore it. This project reads the heartbeat signal itself, flags beats that look wrong, and explains the result in plain language, in English, Hindi and Kannada.

It is a follow-up prototype to our paper, [*A Comprehensive Survey on ECG Signal Graph Interpretation*](https://ouci.dntb.gov.ua/works/4Lyyxzwl) (2022), which reviewed methods for interpreting ECG graphs. Here I built and tested one end to end.

🔗 **[Interactive dashboard](https://claude.ai/artifact/T9MjvwvJFoHW6r76A7ktt2)** · 📖 **[2-minute story walkthrough](https://claude.ai/artifact/BW4FsY36V7kofm39NafJJD)**

> ⚠️ A portfolio prototype built on research data. Not a medical device.

---

## Results at a glance

Tested on the **MIT-BIH Arrhythmia Database**: 44 patients (the 4 recordings from pacemaker patients are left out, as is standard) and **100,733 heartbeats labelled by cardiologists**.

| What was measured | Result |
|---|---|
| Heartbeats found | **98.2%** of 100,733 labelled beats |
| Found beats that were real | **97.6%** |
| Ventricular premature beats caught, trained model, unseen patients | **86.8%**, with **97.0%** of its flags correct |
| False alarms on ventricular beats | cut from **2,128** (simple rules) to **83** (trained model) |
| Atrial premature beats caught | **16–22%**: the main weakness, and the next problem to solve |

**Testing at scale changed the roadmap.** On a single recording (Record 100) the simple early-beat rule caught every early beat with zero false alarms. On 22 patients it had never seen, it caught only 16% of atrial beats. Testing on more patients showed that a result from one recording was misleadingly good.

---

## What the dashboard shows

- **Plain-language verdict**: green, amber or red, with what to do next.
- **Patient view** in English, Hindi and Kannada, and a **doctor view** with heart-rate figures and charts.
- **ECG trace on graph paper** with every beat marked (normal, early or unusual shape).
- **Rhythm map**: each beat's gap before it plotted against the gap after it, which makes irregular beats stand out.
- **Sensitivity sliders**: shows the trade-off between catching problem beats and raising false alarms.
- **44-patient test results**: simple rules compared with the trained model, recording by recording.
- **Product thinking**: target users, the problem, success metric, guardrails and roadmap.

---

## How it works

1. **Clean the signal.** Remove baseline drift and electrical noise from lead II (360 samples per second).
2. **Find each heartbeat.** Detect the R-peak (the tall spike of each beat) with NeuroKit2.
3. **Flag early beats.** A beat is early if the gap before it is under 85% of the median of the previous 8 gaps.
4. **Flag unusual shapes.** Compare each beat with this patient's median beat; a shape match below 0.80 counts as unusual.
5. **Train a model.** A random forest learns from beat timing and shape: 41 features per beat, including gaps before and after, how the gap compares with this patient's usual, and a 24-point outline of the beat's shape.
6. **Explain the result.** Turn heart rate and the share of irregular beats into a message the patient can act on.

### A fair test

The model is evaluated with the **inter-patient split of de Chazal et al. (2004)**: it learns from 22 patients and is scored on 22 *different* patients it has never seen. The cut-off for flagging atrial beats was chosen by cross-validation on the training patients only, so the test patients were never used for tuning.

---

## Product thinking

| | |
|---|---|
| **Who it's for** | People who get an ECG printout but can't see a cardiologist quickly, especially outside big cities, and the general physicians who see them first |
| **Problem** | "Abnormal" tells people nothing about what to do next |
| **North Star metric** | Share of users who correctly say what they will do next after reading the result (proposed target: 80%) |
| **Guardrails** | Dangerous beats missed (aim for none) · false-alarm rate · time to understand the result (under 30 seconds) |
| **Key trade-off** | A miss is worse than a false alarm, but too many alarms erode trust and flood clinics |

**Roadmap**
- **Now:** prototype tested on 44 patients; a model that catches ventricular beats with few false alarms
- **Next:** better atrial-beat detection (P-wave features, a second ECG lead); test the wording with 10 patients in Hindi and Kannada
- **Later:** read a phone photo of a paper ECG; share the summary with a doctor; find a clinical validation partner

---

## Repository structure

| File | What it does |
|---|---|
| `analyze.py` | Two-recording walkthrough (records 100 and 208): detection, flagging and a check against cardiologist labels |
| `export_csv.py` | Writes the CSV files the dashboard reads |
| `sweep.py` | Threshold trade-off: flags, misses and false alarms for each setting |
| `benchmark.py` | Full 44-patient test: simple rules vs trained model on unseen patients |
| `data/` | Record 100 and the Record 208 excerpt used in the walkthrough |
| `out/csv/` | Results used by the dashboard |

## Run it yourself

```bash
pip install wfdb neurokit2 scikit-learn numpy pandas pooch

python analyze.py      # two-record walkthrough
python export_csv.py   # dashboard data files
python sweep.py        # threshold trade-off

# For the full test, download the MIT-BIH database (73.5 MB):
# https://physionet.org/content/mitdb/get-zip/1.0.0/
# and unzip it into ./mitdb_raw/
python benchmark.py    # 44-patient test, about 3 minutes
```

## Tech stack

Python · NeuroKit2 (signal processing) · WFDB (reading ECG files) · scikit-learn (random forest) · pandas · NumPy · D3.js (dashboard charts)

## Limitations

- One ECG lead only. Clinical ECGs use 12.
- MIT-BIH was recorded decades ago and has 48 recordings. It is the standard benchmark, but it doesn't represent today's patients.
- Atrial-beat detection is weak (16–22%).
- The Hindi and Kannada wording hasn't been tested with patients yet.

## Data and credits

- Moody GB, Mark RG. *The impact of the MIT-BIH Arrhythmia Database.* IEEE Eng in Med and Biol 20(3):45–50 (2001). Data from [PhysioNet](https://physionet.org/content/mitdb/1.0.0/), licensed under the Open Data Commons Attribution License v1.0.
- Goldberger AL et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation 101(23):e215–e220 (2000).
- de Chazal P, O'Dwyer M, Reilly RB. *Automatic classification of heartbeats using ECG morphology and heartbeat interval features.* IEEE Trans Biomed Eng 51(7):1196–1206 (2004).

## Author

**Harshitha A Reddy**, co-author of *A Comprehensive Survey on ECG Signal Graph Interpretation* (Advances in Intelligent Systems and Technologies, 2022).
