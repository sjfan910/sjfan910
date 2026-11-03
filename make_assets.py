"""Builds the profile's SVGs: the header panel, the tools strip and the divider, in light and dark.

The header panel is not a drawing. It trains a 2-32-32-3 MLP from scratch in NumPy (manual backprop, no
autograd) on a three-arm spiral, and the shaded cells are the class it actually predicts at each point.
Run: python make_assets.py
"""
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "assets"
SEED = 7
POINTS_PER_CLASS = 90
CLASSES = 3
HIDDEN = 32
EPOCHS = 4000
LEARNING_RATE = 0.5
L2 = 1e-4
GRID_W, GRID_H = 46, 30
WIDTH, HEIGHT = 806, 230
PANEL_X, PANEL_Y, PANEL_W, PANEL_H = 470, 20, 316, 190

THEMES = {
    "light": {"bg": "#ffffff", "ink": "#0f172a", "muted": "#475569", "line": "#e2e8f0",
              "classes": ["#c7e7e5", "#f6dcc4", "#dcd6f2"], "dots": ["#2c7a7b", "#c2410c", "#5b21b6"]},
    "dark": {"bg": "#0d1117", "ink": "#e6edf3", "muted": "#8b949e", "line": "#30363d",
             "classes": ["#123a3c", "#3d2817", "#2a2346"], "dots": ["#4fd1c5", "#fb923c", "#a78bfa"]},
}


def spiral(rng):
    xs, ys = [], []
    for c in range(CLASSES):
        r = np.linspace(0.05, 1.0, POINTS_PER_CLASS)
        t = np.linspace(c * 4.0, (c + 1) * 4.0, POINTS_PER_CLASS) + rng.normal(0, 0.2, POINTS_PER_CLASS)
        xs.append(np.c_[r * np.sin(t), r * np.cos(t)])
        ys.append(np.full(POINTS_PER_CLASS, c))
    return np.vstack(xs), np.concatenate(ys)


def forward(params, x):
    w1, b1, w2, b2, w3, b3 = params
    h1 = np.maximum(0, x @ w1 + b1)
    h2 = np.maximum(0, h1 @ w2 + b2)
    logits = h2 @ w3 + b3
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    return h1, h2, p / p.sum(axis=1, keepdims=True)


def train(x, y, rng):
    sizes = [2, HIDDEN, HIDDEN, CLASSES]
    params = []
    for a, b in zip(sizes, sizes[1:]):
        params += [rng.normal(0, np.sqrt(2 / a), (a, b)), np.zeros(b)]
    n = len(x)
    for _ in range(EPOCHS):
        w1, b1, w2, b2, w3, b3 = params
        h1, h2, p = forward(params, x)
        d3 = p.copy()
        d3[np.arange(n), y] -= 1
        d3 /= n
        d2 = (d3 @ w3.T) * (h2 > 0)
        d1 = (d2 @ w2.T) * (h1 > 0)
        grads = [x.T @ d1 + L2 * w1, d1.sum(0), h1.T @ d2 + L2 * w2, d2.sum(0), h2.T @ d3 + L2 * w3, d3.sum(0)]
        params = [w - LEARNING_RATE * g for w, g in zip(params, grads)]
    accuracy = (forward(params, x)[2].argmax(1) == y).mean()
    return params, accuracy


def header(theme, x, y, params, accuracy):
    t = THEMES[theme]
    cw, ch = PANEL_W / GRID_W, PANEL_H / GRID_H
    gx, gy = np.meshgrid(np.linspace(-1.1, 1.1, GRID_W), np.linspace(1.1, -1.1, GRID_H))
    regions = forward(params, np.c_[gx.ravel(), gy.ravel()])[2].argmax(1).reshape(GRID_H, GRID_W)
    cells = "".join(
        f'<rect x="{PANEL_X + i * cw:.1f}" y="{PANEL_Y + j * ch:.1f}" width="{cw + 0.3:.1f}" height="{ch + 0.3:.1f}" fill="{t["classes"][regions[j, i]]}"/>'
        for j in range(GRID_H) for i in range(GRID_W))
    sx = lambda v: PANEL_X + (v + 1.1) / 2.2 * PANEL_W
    sy = lambda v: PANEL_Y + (1.1 - v) / 2.2 * PANEL_H
    dots = "".join(f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="2.1" fill="{t["dots"][c]}"/>' for (a, b), c in zip(x, y))
    font = "font-family=\"-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif\""
    lines = ["CS @ Georgia Tech · Intelligence &amp; Cybersecurity", "Speech AI research intern @ JD.com",
             "CTO @ LearnToGive · 2 published ML papers"]
    text = "".join(f'<text x="24" y="{118 + k * 26}" fill="{t["muted"]}" font-size="15" {font}>{s}</text>' for k, s in enumerate(lines))
    caption = f"2-{HIDDEN}-{HIDDEN}-3 MLP, NumPy backprop · train acc {accuracy:.1%}"
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">'
            f'<rect width="100%" height="100%" rx="10" fill="{t["bg"]}" stroke="{t["line"]}"/>'
            f'<text x="24" y="74" fill="{t["ink"]}" font-size="40" font-weight="700" {font}>Shijia Fan</text>'
            f'{text}<g>{cells}</g><g>{dots}</g>'
            f'<rect x="{PANEL_X}" y="{PANEL_Y}" width="{PANEL_W}" height="{PANEL_H}" fill="none" stroke="{t["line"]}"/>'
            f'<text x="{PANEL_X + PANEL_W}" y="{PANEL_Y + PANEL_H + 15}" text-anchor="end" fill="{t["muted"]}" font-size="10.5" {font}>{caption}</text>'
            "</svg>")


def tools(theme):
    t = THEMES[theme]
    groups = [("ML", ["PyTorch", "TensorFlow", "NumPy", "Pandas", "scikit-learn"]),
              ("Languages", ["Python", "TypeScript", "Java", "C++", "C", "Rust", "SQL"]),
              ("Web", ["React", "Node.js", "Tailwind", "Three.js", "Flask", "FastAPI"]),
              ("Infra", ["Git", "Linux", "Raspberry Pi", "Vercel", "LaTeX"])]
    font = "font-family=\"-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif\""
    rows, y = [], 22
    for label, items in groups:
        x = 100
        row = f'<text x="0" y="{y + 4}" fill="{t["muted"]}" font-size="12" font-weight="600" {font}>{label}</text>'
        for item in items:
            w = 14 + 7.1 * len(item)
            row += (f'<rect x="{x}" y="{y - 12}" width="{w:.0f}" height="22" rx="11" fill="none" stroke="{t["line"]}"/>'
                    f'<text x="{x + w / 2:.0f}" y="{y + 3}" text-anchor="middle" fill="{t["ink"]}" font-size="12" {font}>{item}</text>')
            x += w + 6
        rows.append(row)
        y += 32
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{y - 8}" viewBox="0 0 {WIDTH} {y - 8}">{"".join(rows)}</svg>'


def rule(theme):
    t = THEMES[theme]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="14" viewBox="0 0 {WIDTH} 14">'
            f'<line x1="0" y1="7" x2="{WIDTH}" y2="7" stroke="{t["line"]}"/>'
            + "".join(f'<circle cx="{WIDTH / 2 + d}" cy="7" r="2.5" fill="{t["dots"][k]}"/>' for k, d in enumerate((-14, 0, 14)))
            + "</svg>")


def main():
    rng = np.random.default_rng(SEED)
    x, y = spiral(rng)
    params, accuracy = train(x, y, rng)
    OUT.mkdir(exist_ok=True)
    for theme in THEMES:
        (OUT / f"header-{theme}.svg").write_text(header(theme, x, y, params, accuracy))
        (OUT / f"tools-{theme}.svg").write_text(tools(theme))
        (OUT / f"rule-{theme}.svg").write_text(rule(theme))
    print(f"train accuracy {accuracy:.3f}")


if __name__ == "__main__":
    main()
