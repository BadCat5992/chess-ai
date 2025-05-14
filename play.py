# play.py
import chess
import chess.engine
import tkinter as tk
from tkinter import simpledialog, messagebox
import torch
import torch.nn as nn
import numpy as np
import os

# === CONFIG ===
STOCKFISH_PATH  = "/usr/games/stockfish"   # ggf. anpassen
CHECKPOINT_FILE = "checkpoint_mixed_neural.pth"
DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNICODE_PIECES = {
    'P': '♙','N': '♘','B': '♗','R': '♖','Q': '♕','K': '♔',
    'p': '♟','n': '♞','b': '♝','r': '♜','q': '♛','k': '♚',
}

# === AlphaZero-Style Input ===
def board_to_tensor(board):
    t = np.zeros((12,8,8), dtype=np.float32)
    for sq, pc in board.piece_map().items():
        r = 7 - (sq // 8); c = sq % 8
        idx = (pc.piece_type - 1) + (0 if pc.color == chess.WHITE else 6)
        t[idx][r][c] = 1
    return torch.tensor(t, device=DEVICE)

# === DQN Model ===
class DQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*8*12, 1024),
            nn.ReLU(),
            nn.Linear(1024, 4096)
        )
    def forward(self, x):
        return self.net(x)

# === Move ↔ Index Helpers ===
def move_to_index(m):  return m.from_square * 64 + m.to_square
def index_to_move(i): return chess.Move(i // 64, i % 64)

# === Minimax mit DQN-Evaluation ===
def dqn_search(board, model, depth):
    if depth <= 0 or board.is_game_over():
        with torch.no_grad():
            qv = model(board_to_tensor(board).unsqueeze(0))[0]
            return qv.max().item(), None
    best_val, best_move = -float('inf'), None
    for m in board.legal_moves:
        nb = board.copy(stack=False)
        nb.push(m)
        val, _ = dqn_search(nb, model, depth-1)
        val = -val
        if val > best_val:
            best_val, best_move = val, m
    return best_val, best_move

# === GUI Mensch vs KI ===
def play_gui(sf_depth, lookahead):
    if not os.path.exists(CHECKPOINT_FILE):
        print("⚠️ Kein Modell gefunden! Bitte erst trainieren mit train.py")
        return

    # Modell laden
    chk = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    model = DQN().to(DEVICE)
    model.load_state_dict(chk["model_state"])
    model.eval()

    board = chess.Board()
    root  = tk.Tk(); root.title("Schach vs KI")
    canvas = tk.Canvas(root, width=480, height=480)
    canvas.pack()
    selected = None

    def draw_board():
        canvas.delete("all")
        for i in range(8):
            for j in range(8):
                col = "#EEE" if (i+j)%2==0 else "#555"
                canvas.create_rectangle(j*60, i*60, j*60+60, i*60+60, fill=col)
                sq = chess.square(j,7-i)
                pc = board.piece_at(sq)
                if pc:
                    sym = UNICODE_PIECES.get(pc.symbol(), pc.symbol())
                    canvas.create_text(j*60+30, i*60+30, text=sym, font=("Arial", 36))

    def ai_move():
        if lookahead > 0:
            _, mv = dqn_search(board, model, lookahead)
            mv = mv or random.choice(list(board.legal_moves))
        else:
            s = board_to_tensor(board).unsqueeze(0)
            with torch.no_grad():
                qv = model(s)[0]
            legal = list(board.legal_moves)
            idxs = [move_to_index(m) for m in legal]
            best = max(idxs, key=lambda x: qv[x].item())
            mv = index_to_move(best) if index_to_move(best) in legal else random.choice(legal)

        # Promotion
        if board.piece_at(mv.from_square).piece_type == chess.PAWN \
           and chess.square_rank(mv.to_square) in [0,7]:
            mv = chess.Move(mv.from_square, mv.to_square, promotion=chess.QUEEN)
        board.push(mv)
        draw_board()
        if board.is_game_over():
            msg = "Unentschieden!" if board.is_stalemate() else "KI hat gewonnen!"
            messagebox.showinfo("Game Over", msg)

    def on_click(evt):
        nonlocal selected
        j,i = evt.x//60, 7-evt.y//60
        sq = chess.square(j,i)
        if selected is None:
            if board.piece_at(sq) and board.piece_at(sq).color == board.turn:
                selected = sq
        else:
            mv = chess.Move(selected, sq)
            if board.piece_at(selected).piece_type == chess.PAWN \
               and chess.square_rank(sq) in [0,7]:
                mv = chess.Move(selected, sq, promotion=chess.QUEEN)
            if mv in board.legal_moves:
                board.push(mv)
                draw_board()
                root.after(100, ai_move)
            selected = None
        draw_board()

    canvas.bind("<Button-1>", on_click)
    draw_board()
    root.mainloop()

if __name__ == "__main__":
    import random
    root = tk.Tk(); root.withdraw()
    depth     = 2
    lookahead = simpledialog.askinteger("Lookahead", "Model Lookahead-Tiefe (0=keine Suche):", minvalue=0)
    root.destroy()
    play_gui(depth, lookahead)
