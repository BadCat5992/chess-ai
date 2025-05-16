import tkinter as tk
from tkinter import messagebox
import chess
import torch
import numpy as np
import random

# === CONFIG ===
CHECKPOINT_FILE = "checkpoint_mixed_neural.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# dynamische Eingabe für Lookahead
try:
    LOOKAHEAD = int(input("Lookahead-Tiefe festlegen (0 = keine Suche): "))
except ValueError:
    LOOKAHEAD = 0
    print("Ungültige Eingabe, Lookahead auf 0 gesetzt.")

# === DQN Model Definition (muss zum trainierten Modell passen) ===
import torch.nn as nn
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

# === Hilfsfunktionen ===
def board_to_tensor(board):
    t = np.zeros((12, 8, 8), dtype=np.float32)
    for sq, pc in board.piece_map().items():
        r = 7 - (sq // 8)
        c = sq % 8
        idx = (pc.piece_type - 1) + (0 if pc.color == chess.WHITE else 6)
        t[idx][r][c] = 1
    return torch.tensor(t, device=DEVICE)

def move_to_index(m):
    return m.from_square * 64 + m.to_square

def piece_value(pc):
    return {
        None: 0.0,
        chess.PAWN: 0.2,
        chess.KNIGHT: 0.3,
        chess.BISHOP: 0.3,
        chess.ROOK: 0.5,
        chess.QUEEN: 0.75,
        chess.KING: 0.0
    }.get(pc.piece_type if pc else None, 0.0)

# === DQN Search für Lookahead ===
def dqn_search(board, model, depth):
    if depth <= 0 or board.is_game_over():
        with torch.no_grad():
            qv = model(board_to_tensor(board).unsqueeze(0))[0]
            return qv.max().item(), None
    best_val, best_move = -float('inf'), None
    for m in board.legal_moves:
        nb = board.copy(stack=False)
        nb.push(m)
        val, _ = dqn_search(nb, model, depth - 1)
        val = -val
        if val > best_val:
            best_val, best_move = val, m
    return best_val, best_move

# === Modell laden ===
model = DQN().to(DEVICE)
ckpt = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()

# === GUI Setup ===
CELL_SIZE = 64
PIECE_UNICODES = {
    chess.PAWN:    {'white':'♙','black':'♟'},
    chess.KNIGHT:  {'white':'♘','black':'♞'},
    chess.BISHOP:  {'white':'♗','black':'♝'},
    chess.ROOK:    {'white':'♖','black':'♜'},
    chess.QUEEN:   {'white':'♕','black':'♛'},
    chess.KING:    {'white':'♔','black':'♚'}
}

class ChessGUI(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.board = chess.Board()
        self.selected_sq = None
        self.canvas = tk.Canvas(self, width=8*CELL_SIZE, height=8*CELL_SIZE)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)
        self.draw_board()
        self.pack()

    def draw_board(self):
        self.canvas.delete("all")
        for r in range(8):
            for c in range(8):
                x0, y0 = c*CELL_SIZE, r*CELL_SIZE
                color = '#F0D9B5' if (r+c)%2==0 else '#B58863'
                self.canvas.create_rectangle(x0, y0, x0+CELL_SIZE, y0+CELL_SIZE, fill=color)
                sq = chess.square(c, 7-r)
                pc = self.board.piece_at(sq)
                if pc:
                    uni = PIECE_UNICODES[pc.piece_type]['white' if pc.color==chess.WHITE else 'black']
                    self.canvas.create_text(x0+CELL_SIZE/2, y0+CELL_SIZE/2, text=uni, font=('Arial', 32))
        if self.selected_sq is not None:
            c = chess.square_file(self.selected_sq)
            r = 7 - chess.square_rank(self.selected_sq)
            x0, y0 = c*CELL_SIZE, r*CELL_SIZE
            self.canvas.create_rectangle(x0, y0, x0+CELL_SIZE, y0+CELL_SIZE, outline='blue', width=3)

    def on_click(self, event):
        c = event.x // CELL_SIZE
        r = 7 - (event.y // CELL_SIZE)
        sq = chess.square(c, r)
        if self.selected_sq is None:
            pc = self.board.piece_at(sq)
            if pc and pc.color == self.board.turn:
                self.selected_sq = sq
        else:
            move = chess.Move(self.selected_sq, sq)
            pc = self.board.piece_at(self.selected_sq)
            if pc and pc.piece_type == chess.PAWN and (chess.square_rank(sq) in [0,7]):
                move = chess.Move(self.selected_sq, sq, promotion=chess.QUEEN)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.selected_sq = None
                self.draw_board()
                if self.board.is_game_over():
                    messagebox.showinfo("Game Over", f"Result: {self.board.result()}")
                else:
                    self.after(100, self.ai_move)
            else:
                self.selected_sq = None
        self.draw_board()

    def ai_move(self):
        _, mv = dqn_search(self.board, model, LOOKAHEAD)
        if mv is None:
            mv = random.choice(list(self.board.legal_moves))
        self.board.push(mv)
        self.draw_board()
        if self.board.is_game_over():
            messagebox.showinfo("Game Over", f"Result: {self.board.result()}")

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Play gegen DQN-KI")
    ChessGUI(root)
    root.mainloop()

