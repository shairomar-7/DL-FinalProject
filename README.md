# Hear It or See It: Robust Audio-Visual Piano Transcription

CS5787 Deep Learning (Cornell Tech, Fall 2026) final project. Omar Shair (os256) and Guillermo Basterra (gb569).

**Goal:** transcribe piano (which keys were pressed, and when) from both the **sound** and the **pianist's hands** (hand-tracking points from a top-down camera), so that transcription keeps working when the audio is noisy, the video is degraded, or both.

This repository currently holds the **milestone** stage: data preprocessing, a pretrained audio-only baseline under a noise sweep, a re-implementation of PianoVAM's rule-based video filter, and a check of how well the hand landmarks localize the played key. The fusion model comes next.

## Files

| File | What it does |
|---|---|
| `Untitled0.ipynb` | **Preprocessing.** Downloads PianoVAM from Hugging Face (audio, hand skeletons, MIDI, TSV labels, metadata; no video), converts audio to 16 kHz mono and hand landmarks to `(frames, 2 hands, 21 joints, xyz)` arrays, and saves them to Google Drive. |
| `02_milestone_experiments.ipynb` | **Milestone experiments** (Colab, GPU). Downloads and preprocesses the 9 test recordings directly from Hugging Face (no Drive needed), then runs: (1) video coverage of ground-truth notes, (2) the audio-only model at clean / 20 / 10 / 5 / 0 / −5 dB white noise, with and without the rule filter, (3) the figure. |
| `avpiano.py` | Helper functions used by the notebook: data loading, white-noise degradation, note-level scoring (`mir_eval`, 50 ms onset tolerance), keyboard homography from the four annotated corners, key positions in white-key units, and the rule-based filter. |
| `results/milestone_full_test/` | Milestone results on the full test split (9 recordings, 42,241 notes): summary and per-recording CSVs, video coverage, and the noise-sweep plot. |
| `requirements.txt` | Python packages. |

## How to run

1. Open `02_milestone_experiments.ipynb` in Google Colab with a GPU runtime (T4 is enough; A100 is faster).
2. Keep `USE_DRIVE = False` to download the test data straight from Hugging Face.
3. Run all cells with `MAX_SEC = 120` for a quick check (first 2 minutes of each recording), then set `MAX_SEC = 0` for the full test set.
4. The last cell zips `results/` for download.

Training splits (for the fusion model) are prepared by the preprocessing notebook.

## Milestone results (full test split, white noise)

Note-onset precision / recall / F1 (%), micro-averaged over 42,241 reference notes.

| Noise | Audio only (P / R / F1) | Audio + rule filter (P / R / F1) |
|---|---|---|
| clean | 96.7 / 96.9 / 96.8 | 97.3 / 96.2 / 96.7 |
| 20 dB | 92.7 / 95.7 / 94.2 | 94.4 / 95.0 / 94.7 |
| 10 dB | 83.0 / 85.8 / 84.4 | 88.2 / 85.2 / 86.7 |
| 5 dB | 80.6 / 65.6 / 72.3 | 87.3 / 65.2 / 74.6 |
| 0 dB | 83.3 / 30.3 / 44.5 | 89.3 / 30.2 / 45.1 |
| −5 dB | 86.5 / 7.9 / 14.5 | 90.6 / 7.8 / 14.4 |

- Noise mostly makes the audio model **miss** notes (recall 30% at 0 dB) rather than invent them (precision stays above 80%).
- The rule filter raises precision but never recovers a missed note (0 in all 54 recording × noise-level runs).
- A fingertip lies within one white key of the played key for 99.2% of test notes.

## Data

[PianoVAM](https://huggingface.co/datasets/PianoVAM/PianoVAM_v1) (Kim et al., ISMIR 2025), released under CC BY-NC-SA 4.0. The data is **not** included in this repository; the notebooks download it.

## References

- Y. Kim et al. *PianoVAM: A Multimodal Piano Performance Dataset.* ISMIR 2025. arXiv:2509.08800
- B. Shi, W.-N. Hsu, K. Lakhotia, A. Mohamed. *Learning Audio-Visual Speech Representation by Masked Multimodal Cluster Prediction (AV-HuBERT).* ICLR 2022. arXiv:2201.02184
- Q. Kong, B. Li, X. Song, Y. Wan, Y. Wang. *High-resolution Piano Transcription with Pedals by Regressing Onset and Offset Times.* arXiv:2010.01815 (pretrained audio model, via the `piano_transcription_inference` package)
- Y. Kim, H. Sohn, J. Nam, A. Lerch. *Multi-Task Multi-Frame Visual Piano Transcription (V2N).* ISMIR 2026. arXiv:2608.03419
- C. Raffel et al. *mir_eval.* ISMIR 2014

## Use of generative AI

Claude (Anthropic) assisted with writing `avpiano.py`, `02_milestone_experiments.ipynb` and this README. All results were produced by running the code ourselves.
