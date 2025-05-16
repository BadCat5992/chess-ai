import chess
import chess.engine
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os

# === CONFIG ===
CHECKPOINT_FILE = "checkpoint_mixed_neural.pth"
STOCKFISH_PATH = "/usr/games/stockfish"    # Pfad zu deiner Stockfish-Binary
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

START_LEVEL = 1
MAX_LEVEL = 3      # Stockfish max Skill Level
WIN_THRESHOLD = 10   # Wins bis Level-Up beim SF-Gegner

print(f"🚀 Running on device: {DEVICE}")

# === AlphaZero-Style Input ===
def board_to_tensor(board):
    t = np.zeros((12, 8, 8), dtype=np.float32)
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
        if len(self.buf) >= self.cap:
            self.buf.pop(0)
        self.buf.append(data)
    def sample(self, bs):
        batch = random.sample(self.buf, bs)
        return tuple(map(list, zip(*batch)))
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

# === Helpers ===
def move_to_index(m): return m.from_square * 64 + m.to_square

def piece_value(pc):
    if pc is None: return 0.0
    return {
        chess.PAWN: 0.2,
        chess.KNIGHT: 0.3,
        chess.BISHOP: 0.3,
        chess.ROOK: 0.5,
        chess.QUEEN: 0.75,
        chess.KING: 0.0
    }.get(pc.piece_type, 0.0)

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
        val, _ = dqn_search(nb, model, depth - 1)
        val = -val
        if val > best_val:
            best_val, best_move = val, m
    return best_val, best_move

# === TRAINING ===
def train(total_episodes, sf_depth, lookahead):
    # Engine starten
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    sf_level = START_LEVEL
    engine.configure({"Skill Level": sf_level})
    sf_wins = 0

    model = DQN().to(DEVICE)
    level = 1  # eigenes Level bleibt, kann optional genutzt werden

    # Checkpoint laden
    start_ep = 1
    if os.path.exists(CHECKPOINT_FILE):
        try:
            chk = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
            model.load_state_dict(chk["model_state"])
            start_ep = chk.get("epoch", 1) + 1
            print(f"✅ Checkpoint geladen! Ab Epoche {start_ep}")
        except Exception as e:
            print("⚠️ Checkpoint inkompatibel, starte neu:", e)

    opt = optim.Adam(model.parameters(), lr=5e-4)
    loss_fn = nn.MSELoss()
    buf, wins, losses_count = ReplayBuffer(), 0, 0

    for ep in range(start_ep, total_episodes + 1):
        board, loss_ep, ply = chess.Board(), 0, 0
        print(f"\n▶️ Episode {ep} vs Stockfish L{sf_level}")

        while not board.is_game_over():
            ply += 1
            s = board_to_tensor(board).unsqueeze(0)

            # KI-Zug
            if lookahead > 0:
                _, mv = dqn_search(board, model, lookahead)
                mv = mv or random.choice(list(board.legal_moves))
            else:
                with torch.no_grad():
                    qv = model(s)[0]
                legal = list(board.legal_moves)
                if random.random() < 0.1:
                    mv = random.choice(legal)
                else:
                    idxs = [move_to_index(m) for m in legal]
                    best_q = -float('inf'); best_mv = None
                    for i, move in zip(idxs, legal):
                        if qv[i].item() > best_q:
                            best_q = qv[i].item(); best_mv = move
                    mv = best_mv or random.choice(legal)

            # Pawn-Promo-Flag
            if board.piece_at(mv.from_square) and board.piece_at(mv.from_square).piece_type == chess.PAWN \
               and chess.square_rank(mv.to_square) in [0, 7]:
                mv = chess.Move(mv.from_square, mv.to_square, promotion=chess.QUEEN)

            cap_pc = board.piece_at(mv.to_square)
            promotion = mv.promotion == chess.QUEEN

            print(f"🤖 KI-Zug {ply}: {board.san(mv)}")
            board.push(mv)

            # === Event-Reward ===
            total_reward = 0.0
            if board.is_checkmate(): total_reward += 1000
            elif board.is_check(): total_reward += 100
            if cap_pc and cap_pc.piece_type == chess.QUEEN: total_reward += 45
            if promotion: total_reward += 50
            if cap_pc: total_reward += piece_value(cap_pc) * 100

            # Gegner-Zug via Stockfish
            if board.is_game_over(): break
            ply += 1
            sf_move = engine.play(board, chess.engine.Limit(depth=sf_depth)).move

            prev_pc = board.piece_at(sf_move.to_square)
            loss_penalty = 0.0
            if prev_pc and prev_pc.piece_type == chess.QUEEN: loss_penalty -= 50
            if prev_pc: loss_penalty -= piece_value(prev_pc) * 100

            print(f"🧠 SF-Zug {ply}: {board.san(sf_move)}")
            board.push(sf_move)

            done = board.is_game_over()
            buf.push(
                s.squeeze(),
                move_to_index(mv),
                total_reward + loss_penalty,
                board_to_tensor(board).squeeze(),
                done
            )

            # Training Step
            if len(buf) > 32:
                b_s, b_a, b_r, b_ns, b_d = buf.sample(32)
                b_s = torch.stack(b_s).to(DEVICE)
                b_a = torch.tensor(b_a, device=DEVICE)
                b_r = torch.tensor(b_r, dtype=torch.float32, device=DEVICE)
                b_ns = torch.stack(b_ns).to(DEVICE)
                b_d = torch.tensor(b_d, dtype=torch.bool, device=DEVICE)

                q = model(b_s).gather(1, b_a.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    qn = model(b_ns).max(1)[0]
                    qn[b_d] = 0.0
                target = b_r + 0.99 * qn

                loss = loss_fn(q, target)
                opt.zero_grad(); loss.backward(); opt.step()
                loss_ep += loss.item()

        # Ergebnis & SF-Level-Up
        res = board.result()
        if res == "1-0":
            sf_wins += 1
            print(f"✅ KI schlägt Stockfish L{sf_level} ({sf_wins}/{WIN_THRESHOLD})")
            if sf_wins >= WIN_THRESHOLD and sf_level < MAX_LEVEL:
                sf_level += 1
                sf_wins = 0
                engine.configure({"Skill Level": sf_level})
                print(f"🔥 Stockfish-Level-Up! Jetzt L{sf_level}")
        else:
            losses_count += 1
            sf_wins = 0
            print(f"✖️ verloren/remis – SF-Level bleibt L{sf_level}")

        print(f"🔢 Episode {ep} abgeschlossen. Stats: SF-Wins@L{sf_level}={sf_wins}, Total Losses={losses_count}")

        # Checkpoint speichern
        torch.save({
            "model_state": model.state_dict(),
            "epoch": ep,
        }, CHECKPOINT_FILE)

    engine.close()

if __name__ == "__main__":
    mode = input("Modus wählen (train / eval): ").strip().lower()
    if mode == "train":
        eps = int(input("Epochen: "))
        depth = int(input("Stockfish-Search-Depth: "))
        lookahead = int(input("Model Lookahead-Tiefe (0=keine Suche): "))
        train(eps, depth, lookahead)
    else:
        print("Eval-Modus noch nicht implementiert")

