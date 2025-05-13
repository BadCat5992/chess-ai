import chess
import chess.engine
import tkinter as tk
from tkinter import simpledialog, messagebox
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import copy
import matplotlib.pyplot as plt

# === CONFIG ===
STOCKFISH_PATH    = "/usr/games/stockfish"   # Pfad anpassen
CHECKPOINT_FILE   = "checkpoint_mixed_neural.pth"
DEVICE            = torch.device("cuda" if torch.cuda.is_available() else "cpu")

START_LEVEL       = 1      # Stockfish-Level 1 (~1350 Elo)
MAX_LEVEL         = 4      # Stockfish-Level 4 (~1700 Elo)
WIN_THRESHOLD     = 10     # Siege bis Level-Up
ELO_STEP_APPROX   = 100    # Anzeige-Elo pro Level

# Unicode-Figuren für GUI
UNICODE_PIECES = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚',
}

# === AlphaZero-Style Input ===
def board_to_tensor(board):
    t = np.zeros((12,8,8), dtype=np.float32)
    for sq, pc in board.piece_map().items():
        r = 7 - (sq // 8); c = sq % 8
        idx = (pc.piece_type - 1) + (0 if pc.color == chess.WHITE else 6)
        t[idx][r][c] = 1
    return torch.tensor(t, device=DEVICE)

# === Replay Buffer ===
class ReplayBuffer:
    def __init__(self, capacity=10000): 
        self.buf, self.cap = [], capacity
    def push(self, *data):
        if len(self.buf) >= self.cap: self.buf.pop(0)
        self.buf.append(data)
    def sample(self, bs): 
        batch = random.sample(self.buf, bs)
        return zip(*batch)
    def __len__(self): 
        return len(self.buf)

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
def move_to_index(m): 
    return m.from_square * 64 + m.to_square
def index_to_move(i): 
    return chess.Move(i // 64, i % 64)

# === Opponents ===
def random_opponent(board): 
    return random.choice(list(board.legal_moves))

def heuristic_opponent(board):
    def value(m):
        pc = board.piece_at(m.to_square)
        return {None:0, chess.PAWN:1, chess.KNIGHT:3, chess.BISHOP:3,
                chess.ROOK:5, chess.QUEEN:9}.get(pc.piece_type if pc else None, 0)
    legal = list(board.legal_moves)
    max_val = max(value(m) for m in legal)
    captures = [m for m in legal if value(m)==max_val and max_val>0]
    return random.choice(captures) if captures else random.choice(legal)

def stockfish_opponent(board, engine, depth):
    return engine.play(board, chess.engine.Limit(depth=depth)).move

# === Minimax mit DQN-Evaluation ===
def dqn_search(board, model, depth):
    if depth <= 0 or board.is_game_over():
        with torch.no_grad():
            qv = model(board_to_tensor(board).unsqueeze(0))[0]
            return qv.max().item(), None
    best_val = -float('inf')
    best_move = None
    for m in board.legal_moves:
        nb = board.copy(stack=False)
        nb.push(m)
        val, _ = dqn_search(nb, model, depth-1)
        val = -val
        if val > best_val:
            best_val, best_move = val, m
    return best_val, best_move

# === Mixed-Opponent Training ===
def train(episodes, sf_depth, lookahead):
    model = DQN().to(DEVICE)
    level = START_LEVEL
    neural_opponents = []

    # **Checkpoint laden, falls vorhanden**
    if os.path.exists(CHECKPOINT_FILE):
        checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state"])
        level = checkpoint.get("level", START_LEVEL)
        print(f"✅ Checkpoint geladen! Starte Training ab Level {level}")

    opt = optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()
    buf = ReplayBuffer()
    losses, wins = [], 0

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"UCI_LimitStrength":True, "UCI_Elo":1300 + level * ELO_STEP_APPROX})

    for ep in range(1, episodes+1):
        board = chess.Board()
        loss_ep = 0

        while not board.is_game_over():
            # — KI-Zug mit optionalem Lookahead —
            if lookahead > 0:
                _, mv = dqn_search(board, model, lookahead)
                if mv is None:
                    mv = random.choice(list(board.legal_moves))
            else:
                s = board_to_tensor(board).unsqueeze(0)
                with torch.no_grad():
                    qv = model(s)[0]
                legal = list(board.legal_moves)
                if random.random() < 0.1:
                    mv = random.choice(legal)
                else:
                    idxs = [move_to_index(m) for m in legal]
                    best = max(idxs, key=lambda i: qv[i].item())
                    mv = index_to_move(best) if index_to_move(best) in legal else random.choice(legal)

            # Promotion prüfen
            if board.piece_at(mv.from_square).piece_type == chess.PAWN \
               and chess.square_rank(mv.to_square) in [0,7]:
                mv = chess.Move(mv.from_square, mv.to_square, promotion=chess.QUEEN)
            board.push(mv)

            # — Gegner-Zug (Mixed) —
            r = random.random()
            if r < 0.3:
                mv2 = random_opponent(board)
            elif r < 0.5:
                mv2 = heuristic_opponent(board)
            elif r < 0.7:
                mv2 = stockfish_opponent(board, engine, sf_depth)
            else:
                if neural_opponents:
                    opp_net = random.choice(neural_opponents)
                    s2 = board_to_tensor(board).unsqueeze(0)
                    with torch.no_grad():
                        qv2 = opp_net(s2)[0]
                    legal2 = list(board.legal_moves)
                    idxs2 = [move_to_index(m) for m in legal2]
                    best2 = max(idxs2, key=lambda i: qv2[i].item())
                    mv2 = index_to_move(best2) if index_to_move(best2) in legal2 else random.choice(legal2)
                else:
                    mv2 = random_opponent(board)

            # Promotion beim Gegner
            if board.piece_at(mv2.from_square).piece_type == chess.PAWN \
               and chess.square_rank(mv2.to_square) in [0,7]:
                mv2 = chess.Move(mv2.from_square, mv2.to_square, promotion=chess.QUEEN)
            if mv2 in board.legal_moves:
                board.push(mv2)

            # — Reward & Buffer —
            done = board.is_game_over()
            reward = 1.0 if done and board.is_checkmate() else (0.5 if done else 0.0)
            ns = board_to_tensor(board).unsqueeze(0)
            buf.push(
                s.squeeze() if lookahead==0 else board_to_tensor(board).squeeze(),
                move_to_index(mv), reward, ns.squeeze(), done
            )

            # — DQN-Update —
            if len(buf) > 200:
                s_b, a_b, r_b, ns_b, d_b = buf.sample(64)
                s_b = torch.stack(list(s_b)).to(DEVICE)
                ns_b = torch.stack(list(ns_b)).to(DEVICE)
                a_b = torch.tensor(list(a_b), device=DEVICE)
                r_b = torch.tensor(list(r_b), device=DEVICE)
                d_b = torch.tensor(list(d_b), device=DEVICE)

                q      = model(s_b)
                next_q = model(ns_b).detach().max(1)[0]
                tgt    = r_b + (1.0 - d_b.float()) * 0.99 * next_q
                cur    = q.gather(1, a_b.unsqueeze(1)).squeeze()

                loss = loss_fn(cur, tgt)
                opt.zero_grad(); loss.backward(); opt.step()
                loss_ep += loss.item()

        losses.append(loss_ep)
        if board.result() == "1-0":
            wins += 1
        print(f"Ep{ep}/{episodes} Loss:{loss_ep:.3f} Wins@Lvl:{wins}/{WIN_THRESHOLD} Level:{level}")

        # — Level-Up —
        if wins >= WIN_THRESHOLD:
            frozen = copy.deepcopy(model).eval().to("cpu")
            neural_opponents.append(frozen)
            wins = 0
            if level < MAX_LEVEL:
                level += 1
                new_elo = 1300 + level * ELO_STEP_APPROX
                engine.configure({"UCI_Elo": new_elo})
                print(f"🎉 Level-Up! Gegnerlevel jetzt {level} (~{new_elo} Elo)")
            else:
                print(f"🔒 Max Level {MAX_LEVEL} erreicht")

        # — Checkpoint —
        if ep % 10 == 0:
            torch.save({
                "model_state": model.state_dict(),
                "level": level
            }, CHECKPOINT_FILE)
            print(f"💾 Checkpoint bei Ep{ep} gespeichert.")

    engine.quit()
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.show()

# === Eval gegen festen Level 4 (~1700 Elo) ===
def evaluate(games, sf_depth, lookahead):
    if not os.path.exists(CHECKPOINT_FILE):
        print("Kein Checkpoint – erst trainieren!")
        return
    data = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    model = DQN().to(DEVICE)
    model.load_state_dict(data["model_state"])
    model.eval()

    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"UCI_LimitStrength":True,
                      "UCI_Elo":1300+MAX_LEVEL*ELO_STEP_APPROX})

    results = {"win":0,"draw":0,"loss":0}
    for i in range(1, games+1):
        board = chess.Board()
        while not board.is_game_over():
            # Model-Zug mit Lookahead/Fallback
            if lookahead > 0:
                _, mv = dqn_search(board, model, lookahead)
                if mv is None: mv = random.choice(list(board.legal_moves))
            else:
                s = board_to_tensor(board).unsqueeze(0)
                with torch.no_grad(): qv = model(s)[0]
                legal = list(board.legal_moves)
                idxs = [move_to_index(m) for m in legal]
                best = max(idxs, key=lambda x: qv[x].item())
                mv = index_to_move(best) if index_to_move(best) in legal else random.choice(legal)

            # Promotion
            if board.piece_at(mv.from_square).piece_type == chess.PAWN \
               and chess.square_rank(mv.to_square) in [0,7]:
                mv = chess.Move(mv.from_square, mv.to_square, promotion=chess.QUEEN)
            board.push(mv)

            # Stockfish-Zug
            res = engine.play(board, chess.engine.Limit(depth=sf_depth))
            mv2 = res.move
            if board.piece_at(mv2.from_square).piece_type == chess.PAWN \
               and chess.square_rank(mv2.to_square) in [0,7]:
                mv2 = chess.Move(mv2.from_square, mv2.to_square, promotion=chess.QUEEN)
            if mv2 in board.legal_moves:
                board.push(mv2)

        r = board.result()
        if r == "1-0":
            results["win"] += 1
        elif r == "1/2-1/2":
            results["draw"] += 1
        else:
            results["loss"] += 1
        print(f"Eval {i}/{games}: {r}")

    engine.quit()
    print(f"Wins:{results['win']}/{games}  Draws:{results['draw']}/{games}  Loss:{results['loss']}/{games}")

# === Mensch vs KI GUI ===
def play_gui(sf_depth, lookahead):
    if not os.path.exists(CHECKPOINT_FILE):
        print("Kein Checkpoint – erst trainieren!")
        return
    data = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
    model = DQN().to(DEVICE)
    model.load_state_dict(data["model_state"])
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

    def on_click(evt):
        nonlocal selected
        j,i = evt.x//60, 7-evt.y//60
        sq = chess.square(j,i)
        if selected is None:
            if board.piece_at(sq) and board.piece_at(sq).color == board.turn:
                selected = sq
        else:
            mv = chess.Move(selected, sq)
            # Promotion?
            if board.piece_at(selected).piece_type == chess.PAWN \
               and chess.square_rank(sq) in [0,7]:
                mv = chess.Move(selected, sq, promotion=chess.QUEEN)
            if mv in board.legal_moves:
                board.push(mv)
                draw_board()
                root.after(100, ai_move)
            selected = None
        draw_board()

    def ai_move():
        if lookahead > 0:
            _, mv = dqn_search(board, model, lookahead)
            if mv is None:
                mv = random.choice(list(board.legal_moves))
        else:
            s = board_to_tensor(board).unsqueeze(0)
            with torch.no_grad(): qv = model(s)[0]
            legal = list(board.legal_moves)
            idxs = [move_to_index(m) for m in legal]
            best = max(idxs, key=lambda x: qv[x].item())
            mv = index_to_move(best) if index_to_move(best) in legal else random.choice(legal)
        if board.piece_at(mv.from_square).piece_type == chess.PAWN \
           and chess.square_rank(mv.to_square) in [0,7]:
            mv = chess.Move(mv.from_square, mv.to_square, promotion=chess.QUEEN)
        board.push(mv)
        draw_board()
        if board.is_game_over():
            msg = "Unentschieden!" if board.is_stalemate() else "KI hat gewonnen!"
            messagebox.showinfo("Game Over", msg)

    canvas.bind("<Button-1>", on_click)
    draw_board()
    root.mainloop()

# === MAIN ===
if __name__ == "__main__":
    mode = input("Modus wählen (train / eval / play): ").strip().lower()
    if mode == "train":
        eps       = int(input("Epochen: "))
        depth     = int(input("Stockfish-Search-Depth (Opponent): "))
        lookahead = int(input("Model Lookahead-Tiefe (0=keine Suche): "))
        train(eps, depth, lookahead)
    elif mode == "eval":
        games     = int(input("Eval-Spiele: "))
        depth     = int(input("Stockfish-Search-Depth (Opponent): "))
        lookahead = int(input("Model Lookahead-Tiefe (0=keine Suche): "))
        evaluate(games, depth, lookahead)
    elif mode == "play":
        root = tk.Tk(); root.withdraw()
        depth     = simpledialog.askinteger("Stockfish-Depth", "Stockfish-Search-Depth (Opponent):", minvalue=1)
        lookahead = simpledialog.askinteger("Lookahead", "Model Lookahead-Tiefe (0=keine Suche):", minvalue=0)
        root.destroy()
        play_gui(depth, lookahead)
    else:
        print("Ungültiger Modus. Eingabe: train / eval / play")
