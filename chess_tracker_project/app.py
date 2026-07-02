"""
Streamlit app - local deployment for the chess move tracker.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from chess_tracker.config import (
    DEFAULT_BOARD_SIZE,
    DEFAULT_PIECE_CONF,
    DEFAULT_PIECE_MODEL_URL,
    DEFAULT_POSE_CONF,
    DEFAULT_POSE_MODEL_URL,
    DEFAULT_STABILITY_SECONDS,
)
from chess_tracker.pipeline import ChessVideoTracker, MoveRecord

st.set_page_config(page_title="Chess Move Tracker", page_icon="\u265f\ufe0f", layout="wide")

# --------------------------------------------------------------------------
# Sidebar - configuration
# --------------------------------------------------------------------------
st.sidebar.title("\u265f\ufe0f Chess Move Tracker")
st.sidebar.caption("YOLO11-based board & piece tracking \u2192 FEN / PGN")

with st.sidebar.expander("Model sources", expanded=False):
    pose_model_path = st.text_input("Board pose model (URL or local path)", value=DEFAULT_POSE_MODEL_URL)
    piece_model_path = st.text_input("Piece detection model (URL or local path)", value=DEFAULT_PIECE_MODEL_URL)

with st.sidebar.expander("Detection settings", expanded=False):
    pose_conf = st.slider("Board corner confidence", 0.1, 0.95, DEFAULT_POSE_CONF, 0.05)
    piece_conf = st.slider("Piece detection confidence", 0.1, 0.95, DEFAULT_PIECE_CONF, 0.05)
    stability_seconds = st.slider(
        "Move stability window (seconds)", 0.3, 4.0, DEFAULT_STABILITY_SECONDS, 0.1,
        help="How long a board state must stay unchanged before a move is confirmed. "
             "Higher = fewer false positives from hands/flicker, but slower to react.",
    )
    frame_stride = st.slider(
        "Process every Nth frame", 1, 10, 1,
        help="Increase to speed up processing on long videos or slower (CPU-only) machines.",
    )
    show_live_preview = st.checkbox("Show live detection preview", value=True)

uploaded_video = st.sidebar.file_uploader(
    "Upload a chess game video", type=["mp4", "mov", "avi", "mkv"]
)
run_button = st.sidebar.button("\u25b6\ufe0f Run tracking", type="primary", disabled=uploaded_video is None)

# --------------------------------------------------------------------------
# Cached model loading (loading YOLO weights is the slow part)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO11 models (first run downloads weights)...")
def load_tracker(pose_path: str, piece_path: str, board_size: int, pose_c: float, piece_c: float, stability: float):
    return ChessVideoTracker(
        pose_model_path=pose_path,
        piece_model_path=piece_path,
        board_size=board_size,
        pose_conf=pose_c,
        piece_conf=piece_c,
        stability_seconds=stability,
    )


# --------------------------------------------------------------------------
# Main area
# --------------------------------------------------------------------------
st.title("Chess Move Tracker")
st.write(
    "Upload a top-down-ish video of a chess game being played. The app localizes the "
    "board, tracks the pieces frame by frame, and reconstructs the game as a sequence "
    "of moves with a FEN snapshot after each one."
)

if "result" not in st.session_state:
    st.session_state.result = None

if uploaded_video is not None and run_button:
    # Persist the upload to a temp file so OpenCV can open it by path
    suffix = Path(uploaded_video.name).suffix or ".mp4"
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_file.write(uploaded_video.read())
    tmp_file.flush()
    video_path = tmp_file.name

    tracker = load_tracker(
        pose_model_path, piece_model_path, DEFAULT_BOARD_SIZE, pose_conf, piece_conf, stability_seconds
    )

    progress_bar = st.progress(0.0, text="Starting...")
    col_preview, col_moves = st.columns([2, 1])
    with col_preview:
        preview_slot = st.empty()
    with col_moves:
        st.subheader("Moves (live)")
        moves_slot = st.empty()

    live_moves: list[MoveRecord] = []
    ui_state = {"last_update": 0.0}

    def on_progress(frame_idx: int, total_frames: int) -> None:
        if total_frames > 0:
            progress_bar.progress(
                min(frame_idx / total_frames, 1.0),
                text=f"Frame {frame_idx}/{total_frames}",
            )
        else:
            progress_bar.progress(0.0, text=f"Frame {frame_idx}")

    def on_frame(annotated_bgr, frame_idx: int, total_frames: int) -> None:
        if not show_live_preview or annotated_bgr is None:
            return
        # Throttle UI updates so Streamlit doesn't choke on redraw rate
        now = time.time()
        if now - ui_state["last_update"] < 0.15:
            return
        ui_state["last_update"] = now
        rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        preview_slot.image(rgb, caption=f"Frame {frame_idx}/{total_frames}", use_container_width=True)

    def on_move(record: MoveRecord) -> None:
        live_moves.append(record)
        df = pd.DataFrame([{"#": m.ply, "Color": m.color, "Move": m.san, "FEN": m.fen} for m in live_moves])
        moves_slot.dataframe(df, use_container_width=True, hide_index=True, height=420)

    with st.spinner("Processing video..."):
        result = tracker.process_video(
            video_path,
            progress_callback=on_progress,
            frame_callback=on_frame if show_live_preview else None,
            move_callback=on_move,
            frame_stride=frame_stride,
        )

    Path(video_path).unlink(missing_ok=True)
    progress_bar.progress(1.0, text="Done!")
    st.session_state.result = result
    st.success(f"Tracking complete \u2014 {len(result['moves'])} move(s) detected.")

# --------------------------------------------------------------------------
# Results section (persists after the run via session_state)
# --------------------------------------------------------------------------
result = st.session_state.result
if result is not None:
    moves = result["moves"]
    st.divider()
    st.subheader("Final result")

    tab_moves, tab_pgn, tab_fen = st.tabs(["Move log", "PGN", "Final FEN"])

    with tab_moves:
        if moves:
            df = pd.DataFrame(
                [{"#": m.ply, "Color": m.color, "From": m.from_square, "To": m.to_square,
                  "Move (SAN)": m.san, "FEN after move": m.fen} for m in moves]
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            json_bytes = df.to_json(orient="records", indent=2).encode("utf-8")
            c1, c2 = st.columns(2)
            c1.download_button("\u2b07\ufe0f Download FEN log (CSV)", csv_bytes, "moves_fen.csv", "text/csv")
            c2.download_button("\u2b07\ufe0f Download FEN log (JSON)", json_bytes, "moves_fen.json", "application/json")
        else:
            st.info("No moves were confirmed in this video.")

    with tab_pgn:
        st.code(result["pgn"], language=None)
        st.download_button("\u2b07\ufe0f Download PGN", result["pgn"].encode("utf-8"), "game.pgn", "application/x-chess-pgn")

    with tab_fen:
        st.code(result["final_fen"], language=None)
        st.caption("FEN of the board's final tracked position.")
        st.link_button(
            "Open final position on Lichess",
            f"https://lichess.org/editor/{result['final_fen'].replace(' ', '_')}",
        )
else:
    st.info("Upload a video and click **Run tracking** in the sidebar to get started.")
