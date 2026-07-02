# Chess Move Tracker

Track a chess game from video, detect every move with two fine-tuned YOLO11
models, and export the game as a per-move FEN log and a PGN file. Includes a
Streamlit app for local, interactive use.

Rebuilt as a standalone project from the original `chess-move-tracking-with-yolo11.ipynb`
notebook — same pipeline, reorganized into reusable modules with no
Kaggle/Colab-specific code.

## How it works

1. **`board_localizer.py`** — a YOLO11-Pose model finds the 4 board corners
   (a1, h1, a8, h8) and locks a perspective-warp so every frame after that is
   mapped to a clean top-down 640x640 board.
2. **YOLO11 piece model** — detects the 12 piece classes + a `Hand` class on
   the warped board each frame.
3. **`state_analyzer.py`** — a small state machine that waits for the board
   state to stay stable for N seconds (filtering out hand occlusion / flicker)
   before confirming a move, and diffs consecutive stable states to infer
   `(from_square, to_square)`.
4. **`pgn_generator.py`** — validates each move with `python-chess`, updates
   the board, and can export a FEN snapshot after every move or the full PGN.
5. **`pipeline.py`** — `ChessVideoTracker` wires all of the above together
   into a single `process_video(...)` call with progress/frame/move callbacks.

## Project layout

```
chess_tracker/
    __init__.py
    config.py            # default model URLs & thresholds
    board_localizer.py   # Phase 1
    state_analyzer.py    # Phase 3
    pgn_generator.py     # Phase 4 (+ FEN builder)
    pipeline.py           # orchestration + CSV/JSON/PGN export helpers
app.py                    # Streamlit UI (local deployment)
main.py                   # CLI, no UI needed
requirements.txt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The two YOLO11 weights (board-pose + piece-detector) are downloaded
automatically from Hugging Face the first time you run the app, and cached
locally by `ultralytics` afterward. A GPU is optional — the pipeline falls
back to CPU automatically (`torch.cuda.is_available()`), just slower.

## Run the Streamlit app (local deployment)

```bash
streamlit run app.py
```

Then in the browser tab that opens:
1. Upload a chess game video (`.mp4`, `.mov`, `.avi`, `.mkv`).
2. (Optional) tweak confidence thresholds / stability window / frame stride
   in the sidebar.
3. Click **Run tracking**.
4. Watch moves appear live, then download the FEN log (CSV/JSON) and the PGN
   from the results tabs.

## Run from the command line (no UI)

```bash
python main.py path/to/game.mp4 --out-dir ./output
```

Writes `game_moves.csv`, `game_moves.json` (per-move FEN log), and `game.pgn`
into `--out-dir`.

Useful flags:
- `--frame-stride N` — process every Nth frame to speed things up on long
  videos or CPU-only machines.
- `--stability SECONDS` — how long a position must hold before a move is
  confirmed (default `1.5`). Lower it for faster-paced games, raise it if
  hand occlusion is causing false move detections.
- `--pose-conf` / `--piece-conf` — detection confidence thresholds.

## Using it as a library

```python
from chess_tracker import ChessVideoTracker

tracker = ChessVideoTracker()  # loads both models once
result = tracker.process_video("game.mp4")

for move in result["moves"]:
    print(move.ply, move.color, move.san, move.fen)

print(result["pgn"])
print("Final FEN:", result["final_fen"])
```

## Notes / tuning tips

- The camera should stay roughly fixed once the board is calibrated; call
  `tracker.localizer.reset()` (done automatically per video) if the camera
  moves between clips.
- If moves are being missed, try lowering `--stability`; if false moves are
  being detected from hand occlusion, raise it.
- This is the same detection/state-machine logic as the original notebook —
  no model retraining is included here, only inference. See the models on
  Hugging Face: `surawut/chess-move-tracking-yolo11`.
