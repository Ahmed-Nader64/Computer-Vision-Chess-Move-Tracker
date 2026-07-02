"""
chess_tracker
=============

A computer-vision pipeline that watches a chess game on video, tracks the
pieces with two fine-tuned YOLO11 models, and reconstructs the game as a
sequence of FEN positions / a PGN file.

Modules:
    config           - default model URLs & pipeline constants
    board_localizer  - locates the board and produces a top-down warp
    state_analyzer    - turns per-frame detections into confirmed moves
    pgn_generator    - validates moves and builds FEN/PGN output
    pipeline         - orchestrates the full video -> moves pipeline
"""

from .pipeline import ChessVideoTracker, MoveRecord

__all__ = ["ChessVideoTracker", "MoveRecord"]
