import streamlit as st
import numpy as np
import pandas as pd
import copy, math

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VAM + MODI Solver | MBG Sumedang",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #1E2D25 !important; }
[data-testid="stSidebar"] * { color: #cde8d9 !important; }
[data-testid="stSidebar"] label { color: #7A9B8A !important; font-size:0.75rem; text-transform:uppercase; letter-spacing:.06em; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #1E2D25 0%, #0f3324 100%);
    border-radius: 16px; padding: 2.2rem 2.6rem; margin-bottom: 1.8rem;
    position: relative; overflow: hidden;
}
.hero::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:220px; height:220px; border-radius:50%;
    background: radial-gradient(circle, rgba(29,184,122,.25), transparent 70%);
}
.hero-tag { display:inline-block; background:#1DB87A; color:white; font-size:.7rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; padding:3px 10px; border-radius:20px; margin-bottom:.7rem; }
.hero h1 { color:white; font-size:1.9rem; font-weight:800; margin:0 0 .3rem; }
.hero p { color:#7A9B8A; margin:0; font-size:.92rem; }

/* ── Section labels ── */
.sec-label { font-size:.7rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:#1DB87A; margin-bottom:.4rem; }
.sec-title { font-size:1.3rem; font-weight:700; color:#0D1F17; margin-bottom:1rem; }

/* ── Metric cards ── */
.mcard { background:white; border:1.5px solid #D1E8DB; border-radius:12px; padding:1rem 1.3rem; }
.mcard .ml { font-size:.7rem; color:#7A9B8A; font-weight:600; letter-spacing:.06em; text-transform:uppercase; }
.mcard .mv { font-size:1.65rem; font-weight:800; color:#1DB87A; line-height:1.1; }
.mcard .ms { font-size:.76rem; color:#7A9B8A; }

/* ── Info / warn boxes ── */
.ibox { background:#E8FBF2; border-left:4px solid #1DB87A; border-radius:0 8px 8px 0; padding:.75rem 1rem; margin:.7rem 0; font-size:.86rem; color:#1E2D25; }
.wbox { background:#fff8ec; border-left:4px solid #F5A623; border-radius:0 8px 8px 0; padding:.75rem 1rem; margin:.7rem 0; font-size:.86rem; color:#7a5800; }

/* ── Transport Table ── */
.tp-wrap { overflow-x: auto; padding-bottom: .5rem; }
table.tp {
    border-collapse: collapse;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: .88rem;
    margin: 0 auto;
    white-space: nowrap;
}
table.tp th, table.tp td {
    border: 1.5px solid #cbd5e1;
    min-width: 80px;
    height: 64px;
    text-align: center;
    vertical-align: middle;
    position: relative;
    padding: 2px;
}
table.tp th { background: #f1f5f9; color:#475569; font-weight:600; font-size:.8rem; }

/* Cell types */
.cell-alloc   { background: white; }
.cell-striked { background: #eff6ff; }
.cell-cbar    { background: white; }
.cell-active  { background: #eff6ff; border-color:#818cf8 !important; box-shadow:inset 0 0 0 2px #818cf8; z-index:5; }
.cell-enter   { background: #fefce8; border-color:#eab308 !important; box-shadow:inset 0 0 0 2px #eab308; z-index:5; }
.cell-loop    { background: #eff6ff; border-color:#6366f1 !important; }
.cell-inactive{ background: #e2e8f0; color: #94a3b8; }

.cost-badge {
    position:absolute; top:3px; right:3px;
    background:#f1f5f9; border:1px solid #cbd5e1; border-radius:4px;
    font-size:.68rem; font-weight:700; color:#475569;
    padding:1px 4px; line-height:1.4;
}
.alloc-val {
    font-size:1.2rem; font-weight:700; color:#1e40af;
    display:flex; align-items:center; justify-content:center;
    width:100%; height:100%;
}
.alloc-val.highlighted { color:#3730a3; }
.cbar-val {
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    font-size:.82rem; font-weight:700; width:100%; height:100%;
}
.cbar-pos { color:#16a34a; }
.cbar-neg { color:#dc2626; }

/* Theta shift display */
.theta-shift {
    display:flex; flex-direction:column; align-items:center; justify-content:center;
    width:100%; height:100%;
}
.theta-old { text-decoration:line-through; color:#f87171; font-size:.78rem; font-weight:700; }
.theta-new { color:#16a34a; font-size:1.1rem; font-weight:700; }

/* Supply / Demand cells */
.cell-supply { background:#eff6ff; font-weight:700; color:#1d4ed8; font-size:1rem; }
.cell-demand { background:#f0fdf4; font-weight:700; color:#15803d; font-size:1rem; }
.cell-supply.inactive { background:#e2e8f0; color:#94a3b8; }
.cell-demand.inactive { background:#e2e8f0; color:#94a3b8; }
.cell-total  { background:#1e293b; color:white; font-weight:700; font-size:1rem; }

/* Penalty cells */
.cell-pen { background:#faf5ff; color:#6b21a8; font-size:.9rem; }
.cell-pen-active { background:#ede9fe; color:#4c1d95; font-weight:700; border:2px solid #7c3aed !important; }
.pen-detail { font-size:.62rem; color:#a78bfa; }

/* UV cells */
.cell-u { background:#e0e7ff; color:#3730a3; font-weight:700; font-size:1rem; }
.cell-v { background:#e0e7ff; color:#3730a3; font-weight:700; font-size:1rem; }

/* ── Loop SVG overlay ── */
.step-grid-wrapper { position:relative; display:inline-block; }
.loop-svg { position:absolute; top:0; left:0; pointer-events:none; z-index:20; overflow:visible; }

/* ── Navigation ── */
.nav-bar {
    display:flex; align-items:center; justify-content:center;
    gap:1.2rem; padding:1.2rem;
    border-top:1px solid #e2e8f0; margin-top:1rem;
}
.nav-btn {
    width:44px; height:44px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    background:white; border:1px solid #cbd5e1;
    font-size:1.2rem; cursor:pointer;
    transition:background .15s;
}
.nav-btn:hover { background:#f1f5f9; }
.nav-btn:disabled { opacity:.35; cursor:not-allowed; }
.dots { display:flex; gap:5px; align-items:center; flex-wrap:wrap; justify-content:center; max-width:320px; }
.dot { width:10px; height:10px; border-radius:5px; background:#cbd5e1; transition:all .25s; }
.dot.active { width:28px; background:#4f46e5; }

/* ── Step description box ── */
.step-desc {
    background:white; border:1.5px solid #e0e7ff;
    border-radius:12px; padding:1rem 1.2rem; margin:1rem 0;
    font-size:.88rem; color:#1e293b;
}

/* ── Result banner ── */
.result-banner {
    background:linear-gradient(135deg,#1DB87A,#17a36b);
    color:white; border-radius:14px;
    padding:1.8rem 2.2rem; text-align:center; margin-top:1.5rem;
}
.result-banner h2 { font-size:1rem; font-weight:600; margin:0 0 .25rem; opacity:.85; }
.big-cost { font-size:2.8rem; font-weight:800; letter-spacing:-1px; }

/* ── Button ── */
.stButton > button {
    background:#4f46e5; color:white; border:none;
    border-radius:8px; font-weight:600; font-size:.9rem;
    padding:.45rem 1.3rem; transition:background .2s;
}
.stButton > button:hover { background:#4338ca; color:white; }

#MainMenu {visibility:hidden;} footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  ALGORITHM: VAM
# ═══════════════════════════════════════════════
def vam_solve(cost, supply, demand, row_names, col_names):
    """Returns list of step dicts with full penalty history."""
    R, C = len(supply), len(demand)
    sup = list(supply); dem = list(demand)
    alloc = [[0]*C for _ in range(R)]
    steps = []
    history_r = []   # list of {row_idx: {val, detail}}
    history_c = []   # list of {col_idx: {val, detail}}
    active_r = set(range(R))
    active_c = set(range(C))

    def pen(vals):
        s = sorted(vals)
        if len(s) >= 2: return s[1]-s[0], f"{s[1]}-{s[0]}"
        if len(s) == 1: return s[0], str(s[0])
        return -1, "-"

    it = 0
    steps.append({
        "desc": "Mulai iterasi VAM. Hitung penalti (selisih 2 biaya terendah) untuk setiap baris dan kolom aktif.",
        "alloc": [r[:] for r in alloc],
        "active_r": set(active_r), "active_c": set(active_c),
        "highlight": None,
        "history_r": [dict(h) for h in history_r],
        "history_c": [dict(h) for h in history_c],
        "pen_r": {}, "pen_c": {},
        "target_pen": None,
    })

    while active_r and active_c:
        it += 1
        # compute penalties
        pr = {}; pc = {}
        for i in active_r:
            vals = [cost[i][j] for j in active_c]
            v, d = pen(vals)
            pr[i] = {"val": v, "detail": d}
        for j in active_c:
            vals = [cost[i][j] for i in active_r]
            v, d = pen(vals)
            pc[j] = {"val": v, "detail": d}

        # find max penalty
        max_v = -1
        is_row = True; idx = -1
        for i, p in pr.items():
            if p["val"] > max_v: max_v = p["val"]; is_row = True; idx = i
        for j, p in pc.items():
            if p["val"] > max_v: max_v = p["val"]; is_row = False; idx = j

        # find min cost cell
        if is_row:
            min_c = min(active_c, key=lambda j: cost[idx][j])
            ai, aj = idx, min_c
        else:
            min_r = min(active_r, key=lambda i: cost[i][idx])
            ai, aj = min_r, idx

        qty = min(sup[ai], dem[aj])
        alloc[ai][aj] += qty
        sup[ai] -= qty; dem[aj] -= qty

        snap_hr = [dict(h) for h in history_r] + [dict(pr)]
        snap_hc = [dict(h) for h in history_c] + [dict(pc)]

        desc_parts = [
            f"<b>Iterasi {it}:</b> Penalti terbesar = <b>{max_v}</b> pada {'Baris' if is_row else 'Kolom'} <b>{(row_names+col_names)[ai if is_row else R+aj]}</b>.",
            f"Biaya terkecil di {'baris' if is_row else 'kolom'} tersebut = <b>{cost[ai][aj]}</b> di sel <b>({row_names[ai]}, {col_names[aj]})</b>.",
            f"Alokasikan <b>{qty}</b> unit.",
            f"{'Supply '+row_names[ai]+' habis → baris dicoret.' if sup[ai]==0 else ''} "
            f"{'Demand '+col_names[aj]+' terpenuhi → kolom dicoret.' if dem[aj]==0 else ''}",
        ]

        if sup[ai] == 0: active_r.discard(ai)
        if dem[aj] == 0: active_c.discard(aj)

        history_r.append(dict(pr))
        history_c.append(dict(pc))

        steps.append({
            "desc": " ".join(desc_parts),
            "alloc": [r[:] for r in alloc],
            "active_r": set(active_r), "active_c": set(active_c),
            "highlight": (ai, aj),
            "history_r": [dict(h) for h in history_r],
            "history_c": [dict(h) for h in history_c],
            "pen_r": dict(pr), "pen_c": dict(pc),
            "target_pen": {"is_row": is_row, "idx": idx, "it": it-1},
        })

    return alloc, steps


# ═══════════════════════════════════════════════
#  ALGORITHM: MODI
# ═══════════════════════════════════════════════
def get_basic(alloc, R, C):
    return [(i,j) for i in range(R) for j in range(C) if alloc[i][j] is not None and alloc[i][j] > 0]

def compute_uv(cost, alloc, R, C):
    basic = get_basic(alloc, R, C)
    u = [None]*R; v = [None]*C
    # init: row with most allocations
    counts = [sum(1 for j in range(C) if alloc[i][j] is not None and alloc[i][j]>0) for i in range(R)]
    u[counts.index(max(counts))] = 0
    changed = True
    while changed:
        changed = False
        for (i,j) in basic:
            if u[i] is not None and v[j] is None:
                v[j] = cost[i][j] - u[i]; changed = True
            elif v[j] is not None and u[i] is None:
                u[i] = cost[i][j] - v[j]; changed = True
    return u, v

def find_loop(alloc, ei, ej, R, C):
    """Find closed loop for entering cell (ei,ej) using DFS."""
    basic_cells = get_basic(alloc, R, C)
    all_cells = [(ei, ej)] + basic_cells

    def backtrack(path, need_row):
        curr = path[-1]
        if len(path) > 3:
            if need_row and curr[0] == ei:
                return path
            if not need_row and curr[1] == ej:
                return path
        if len(path) > 2*(R+C): return None
        for cell in all_cells:
            if cell in path[1:]: continue
            if need_row and cell[0] == curr[0] and cell[1] != curr[1]:
                r = backtrack(path+[cell], False)
                if r: return r
            elif not need_row and cell[1] == curr[1] and cell[0] != curr[0]:
                r = backtrack(path+[cell], True)
                if r: return r
        return None

    for cell in all_cells[1:]:
        if cell[0] == ei:
            r = backtrack([(ei,ej), cell], False)
            if r and r[-1][0] == ei: return r
    return None

def modi_solve(cost, initial_alloc, row_names, col_names):
    R, C = len(row_names), len(col_names)
    alloc = [[v if v>0 else None for v in row] for row in initial_alloc]
    # degeneracy fix
    basic_count = len(get_basic(alloc,R,C))
    req = R+C-1
    if basic_count < req:
        for i in range(R):
            for j in range(C):
                if alloc[i][j] is None and basic_count < req:
                    alloc[i][j] = 0; basic_count += 1

    steps = []

    def tc():
        s = 0
        for i in range(R):
            for j in range(C):
                v = alloc[i][j]
                if v is not None: s += v * cost[i][j]
        return s

    it = 0
    max_it = 30
    while it < max_it:
        it += 1
        u, v = compute_uv(cost, alloc, R, C)

        # compute C̄ for non-basic
        cbar = {}
        for i in range(R):
            for j in range(C):
                if (alloc[i][j] is None or alloc[i][j]==0) and alloc[i][j] is None:
                    if u[i] is not None and v[j] is not None:
                        cbar[(i,j)] = cost[i][j] - u[i] - v[j]

        # step 1: show u,v
        steps.append({
            "phase": "uv",
            "it": it,
            "alloc": [r[:] for r in alloc],
            "u": list(u), "v": list(v),
            "cbar": {},
            "highlight": None, "loop": None, "action_cells": None,
            "total_cost": tc(),
            "desc": (
                f"<b>Iterasi {it} – Tahap 1: Hitung u<sub>i</sub> & v<sub>j</sub></b><br>"
                f"Gunakan rumus <b>C<sub>ij</sub> = u<sub>i</sub> + v<sub>j</sub></b> pada sel-sel basic (terisi).<br>"
                f"Inisialisasi u<sub>{list(u).index(0)+1 if 0 in u else '?'}</sub> = 0 (baris dengan alokasi terbanyak).<br>"
                f"<span style='font-family:monospace'>u: [{', '.join(str(x) if x is not None else '–' for x in u)}]</span><br>"
                f"<span style='font-family:monospace'>v: [{', '.join(str(x) if x is not None else '–' for x in v)}]</span>"
            ),
            "optimal": False,
        })

        # step 2: show C̄
        steps.append({
            "phase": "cbar",
            "it": it,
            "alloc": [r[:] for r in alloc],
            "u": list(u), "v": list(v),
            "cbar": dict(cbar),
            "highlight": None, "loop": None, "action_cells": None,
            "total_cost": tc(),
            "desc": (
                f"<b>Iterasi {it} – Tahap 2: Evaluasi Sel Kosong (C̄<sub>ij</sub>)</b><br>"
                f"Rumus: <b>C̄<sub>ij</sub> = C<sub>ij</sub> – u<sub>i</sub> – v<sub>j</sub></b>. "
                f"Nilai C̄ langsung ditampilkan di dalam sel kosong.<br>"
                f"Jika semua C̄ ≥ 0 → <span style='color:#16a34a;font-weight:700'>Solusi sudah OPTIMAL</span>."
            ),
            "optimal": False,
        })

        # check optimality
        neg = {k: vv for k, vv in cbar.items() if vv < 0}
        if not neg:
            steps.append({
                "phase": "optimal",
                "it": it,
                "alloc": [r[:] for r in alloc],
                "u": list(u), "v": list(v),
                "cbar": dict(cbar),
                "highlight": None, "loop": None, "action_cells": None,
                "total_cost": tc(),
                "desc": (
                    f"<div style='background:#f0fdf4;border:1.5px solid #86efac;border-radius:10px;padding:.8rem 1rem;color:#14532d;font-weight:600;text-align:center'>"
                    f"✅ Semua nilai C̄<sub>ij</sub> ≥ 0 — Solusi sudah <b>OPTIMAL</b>! Total biaya = <b>{tc():,.0f}</b>"
                    f"</div>"
                ),
                "optimal": True,
            })
            break

        # pick most negative
        ei, ej = min(neg, key=lambda k: neg[k])
        min_val = neg[(ei,ej)]

        # find loop
        loop = find_loop(alloc, ei, ej, R, C)
        if loop is None:
            steps.append({"phase": "error", "it": it, "alloc":[r[:] for r in alloc],
                "u": list(u), "v": list(v), "cbar": dict(cbar),
                "highlight": None, "loop": None, "action_cells": None,
                "total_cost": tc(), "desc": "Loop tidak ditemukan (degenerasi).", "optimal": True})
            break

        signed = [(cell, '+' if k%2==0 else '-') for k, cell in enumerate(loop)]
        minus_cells = [(c, s) for c, s in signed if s=='-']
        theta = min(alloc[c[0]][c[1]] for c, s in minus_cells)

        action_cells = {}
        for cell, sign in signed:
            old = alloc[cell[0]][cell[1]] or 0
            new = old + theta if sign=='+' else old - theta
            action_cells[cell] = {"old": old, "new": new, "sign": sign}

        shift_lines = "".join(
            f"<li>Sel ({r_+1},{c_+1}) [{row_names[r_]}→{col_names[c_]}]: "
            f"<span style='text-decoration:line-through;color:#f87171'>{action_cells[(r_,c_)]['old']}</span> "
            f"{'+ ' if sign=='+'  else '– '}{theta} = "
            f"<b style='color:{'#166534' if action_cells[(r_,c_)]['new']>0 else '#94a3b8'}'>"
            f"{action_cells[(r_,c_)]['new'] or '0 (keluar)'}</b></li>"
            for (r_,c_), sign in signed
        )

        steps.append({
            "phase": "shift",
            "it": it,
            "alloc": [r[:] for r in alloc],
            "u": list(u), "v": list(v),
            "cbar": {},
            "highlight": (ei, ej),
            "loop": signed,
            "action_cells": action_cells,
            "total_cost": tc(),
            "desc": (
                f"<b>Iterasi {it} – Tahap 3: Pergeseran Alokasi</b><br>"
                f"Sel masuk: <b>({row_names[ei]}, {col_names[ej]})</b> karena C̄ = <b style='color:#dc2626'>{min_val}</b>.<br>"
                f"θ = <b>{theta}</b> (nilai terkecil pada sel bertanda −).<br>"
                f"<ul style='margin:.5rem 0 0 1rem;font-size:.85rem'>{shift_lines}</ul>"
            ),
            "optimal": False,
        })

        # apply shift
        for cell, sign in signed:
            old = alloc[cell[0]][cell[1]] or 0
            alloc[cell[0]][cell[1]] = old + theta if sign=='+' else old - theta
        for cell, _ in minus_cells:
            if alloc[cell[0]][cell[1]] == 0:
                alloc[cell[0]][cell[1]] = None

    return alloc, steps


# ═══════════════════════════════════════════════
#  HTML TABLE RENDERER
# ═══════════════════════════════════════════════
def render_transport_table(
    cost, alloc, supply, demand,
    row_names, col_names,
    active_r=None, active_c=None, highlight=None,
    pen_r=None, pen_c=None,
    history_r=None, history_c=None,
    target_pen=None,
    u_vals=None, v_vals=None,
    cbar=None, loop=None, action_cells=None,
    show_penalties=True,
):
    R, C = len(row_names), len(col_names)
    if active_r is None: active_r = set(range(R))
    if active_c is None: active_c = set(range(C))
    if history_r is None: history_r = []
    if history_c is None: history_c = []
    if cbar is None: cbar = {}
    if loop is None: loop = []
    if action_cells is None: action_cells = {}

    loop_cells = {cell: sign for cell, sign in loop}
    n_pen = len(history_r)
    show_uv = u_vals is not None

    html = ['<div class="tp-wrap"><table class="tp">']

    # ── HEADER ROW ──
    html.append('<thead><tr>')
    if show_uv:
        html.append('<th style="background:#e0e7ff;color:#3730a3">u<sub>i</sub></th>')
    html.append('<th style="min-width:110px">SPPG \\ Sekolah</th>')
    for j, name in enumerate(col_names):
        inactive = j not in active_c
        cls = 'style="background:#e2e8f0;color:#94a3b8"' if inactive else ''
        html.append(f'<th {cls}>{name}</th>')
    html.append('<th style="background:#dbeafe;color:#1e40af">Supply</th>')
    if show_penalties and n_pen > 0:
        for k in range(n_pen):
            is_target = target_pen and target_pen.get("is_row") is not None and (k == target_pen.get("it", -1))
            cls = 'style="background:#ede9fe;color:#4c1d95;border:2px solid #7c3aed"' if is_target else 'style="background:#faf5ff;color:#6b21a8"'
            html.append(f'<th {cls} title="Penalti iterasi {k+1}">P{k+1}</th>')
    html.append('</tr></thead><tbody>')

    # ── DATA ROWS ──
    for i, rname in enumerate(row_names):
        inactive_row = i not in active_r
        html.append('<tr>')

        # u_i
        if show_uv:
            uv = u_vals[i]
            html.append(f'<td class="cell-u">{uv if uv is not None else "–"}</td>')

        # row label
        label_cls = 'style="background:#e2e8f0;color:#94a3b8"' if inactive_row else ''
        html.append(f'<th {label_cls}>{rname}</th>')

        # data cells
        for j in range(C):
            inactive_col = j not in active_c
            cell_key = (i, j)
            is_highlight = highlight == (i, j)
            in_loop = cell_key in loop_cells
            in_action = cell_key in action_cells

            # determine cell class
            if inactive_row or inactive_col:
                td_cls = 'cell-inactive'
            elif is_highlight:
                td_cls = 'cell-enter'
            elif in_loop and not in_action:
                td_cls = 'cell-loop'
            else:
                td_cls = 'cell-alloc'

            c_badge = cost[i][j]
            html.append(f'<td class="{td_cls}">')
            html.append(f'<span class="cost-badge">{c_badge}</span>')

            # cell content
            if in_action:
                d = action_cells[cell_key]
                sign_sym = '+' if d['sign']=='+' else '−'
                html.append('<div class="theta-shift">')
                html.append(f'<span class="theta-old">{d["old"]}</span>')
                if d['new'] > 0:
                    html.append(f'<span class="theta-new">{d["new"]}</span>')
                html.append('</div>')
            elif cell_key in cbar:
                val = cbar[cell_key]
                cls2 = 'cbar-neg' if val < 0 else 'cbar-pos'
                html.append(f'<div class="cbar-val"><span class="{cls2}" style="text-decoration:overline">C</span><span class="{cls2}">{val:+}</span></div>')
            else:
                v = alloc[i][j]
                if v is None: v = 0
                alloc_cls = 'alloc-val highlighted' if in_loop or is_highlight else 'alloc-val'
                html.append(f'<div class="{alloc_cls}">{v if v > 0 else ""}</div>')

            html.append('</td>')

        # supply
        sup_cls = 'cell-supply inactive' if inactive_row else 'cell-supply'
        html.append(f'<td class="{sup_cls}">{supply[i]}</td>')

        # penalty columns
        if show_penalties:
            for k, hr in enumerate(history_r):
                is_target = (target_pen and target_pen.get("is_row") and
                             target_pen.get("idx") == i and target_pen.get("it") == k)
                pdata = hr.get(i, {})
                pen_cls = 'cell-pen-active' if is_target else 'cell-pen'
                if pdata:
                    html.append(f'<td class="{pen_cls}">')
                    html.append(f'<div>{pdata.get("val","–")}</div>')
                    html.append(f'<div class="pen-detail">({pdata.get("detail","–")})</div>')
                    html.append('</td>')
                else:
                    html.append(f'<td class="{pen_cls}">–</td>')

        html.append('</tr>')

    # ── DEMAND ROW ──
    html.append('<tr>')
    if show_uv:
        html.append('<td style="border:none;background:transparent"></td>')
    html.append('<th style="background:#dcfce7;color:#15803d">Demand</th>')
    for j in range(C):
        inactive = j not in active_c
        dem_cls = 'cell-demand inactive' if inactive else 'cell-demand'
        html.append(f'<td class="{dem_cls}">{demand[j]}</td>')
    total_s = sum(supply)
    html.append(f'<td class="cell-total">{total_s}</td>')
    if show_penalties and n_pen > 0:
        html.append(f'<td colspan="{n_pen}" style="background:#1e293b"></td>')
    html.append('</tr>')

    # ── COLUMN PENALTY ROWS ──
    if show_penalties:
        for k, hc in enumerate(history_c):
            is_target_row = target_pen and not target_pen.get("is_row") and target_pen.get("it") == k
            html.append('<tr>')
            if show_uv:
                html.append('<td style="border:none;background:transparent"></td>')
            html.append(f'<th style="background:#faf5ff;color:#6b21a8;font-size:.75rem;text-align:right;padding-right:.5rem">P{k+1} Kolom</th>')
            for j in range(C):
                is_target = is_target_row and target_pen.get("idx") == j
                pdata = hc.get(j, {})
                pen_cls = 'cell-pen-active' if is_target else 'cell-pen'
                if pdata:
                    html.append(f'<td class="{pen_cls}">')
                    html.append(f'<div>{pdata.get("val","–")}</div>')
                    html.append(f'<div class="pen-detail">({pdata.get("detail","–")})</div>')
                    html.append('</td>')
                else:
                    html.append(f'<td class="{pen_cls}">–</td>')
            html.append(f'<td style="background:#1e293b"></td>')
            if n_pen > 0:
                html.append(f'<td colspan="{n_pen}" style="background:#1e293b"></td>')
            html.append('</tr>')

    # ── v_j ROW ──
    if show_uv and v_vals is not None:
        html.append('<tr>')
        html.append('<td style="border:none;background:transparent"></td>')
        html.append('<th style="background:#e0e7ff;color:#3730a3;font-size:.75rem">v<sub>j</sub></th>')
        for j in range(C):
            vv = v_vals[j]
            html.append(f'<td class="cell-v">{vv if vv is not None else "–"}</td>')
        html.append('<td style="background:#e0e7ff"></td>')
        html.append('</tr>')

    html.append('</tbody></table></div>')
    return ''.join(html)


# ═══════════════════════════════════════════════
#  SESSION STATE INIT
# ═══════════════════════════════════════════════
def init_state():
    defaults = {
        "n_sppg": 2, "n_sek": 6,
        "sppg_names": ["SPPG Awisurat Tanjungsari", "SPPG Cinanjung Tanjungsari"],
        "sek_names": ["SDN Ciluluk II","SMPN 2 Tanjungsari","SDN Maruyung I","SDN Sukamantri","SDN Tanjungsari III","SDN Cikandang"],
        "supply": [2379, 2946],
        "demand": [506, 1075, 406, 404, 226, 498],
        "cost": [[1600,26,1700,900,700,2300],[2900,1300,5000,400,2000,4000]],
        "solved": False,
        "vam_steps": None, "modi_steps": None,
        "vam_alloc": None, "modi_alloc": None,
        "vam_step_idx": 0, "modi_step_idx": 0,
        "cost_s": None, "sup_s": None, "dem_s": None,
        "rnames_s": None, "cnames_s": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ═══════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Konfigurasi")
    st.markdown("---")
    new_ns = st.slider("Jumlah SPPG", 1, 8, st.session_state.n_sppg)
    new_nk = st.slider("Jumlah Sekolah", 2, 12, st.session_state.n_sek)

    if new_ns != st.session_state.n_sppg or new_nk != st.session_state.n_sek:
        st.session_state.n_sppg = new_ns; st.session_state.n_sek = new_nk
        while len(st.session_state.sppg_names) < new_ns: st.session_state.sppg_names.append(f"SPPG {len(st.session_state.sppg_names)+1}")
        st.session_state.sppg_names = st.session_state.sppg_names[:new_ns]
        while len(st.session_state.sek_names) < new_nk: st.session_state.sek_names.append(f"Sekolah {len(st.session_state.sek_names)+1}")
        st.session_state.sek_names = st.session_state.sek_names[:new_nk]
        while len(st.session_state.supply) < new_ns: st.session_state.supply.append(1000)
        st.session_state.supply = st.session_state.supply[:new_ns]
        while len(st.session_state.demand) < new_nk: st.session_state.demand.append(300)
        st.session_state.demand = st.session_state.demand[:new_nk]
        while len(st.session_state.cost) < new_ns: st.session_state.cost.append([500]*new_nk)
        st.session_state.cost = st.session_state.cost[:new_ns]
        for i in range(new_ns):
            while len(st.session_state.cost[i]) < new_nk: st.session_state.cost[i].append(500)
            st.session_state.cost[i] = st.session_state.cost[i][:new_nk]
        st.session_state.solved = False; st.rerun()

    st.markdown("### 📌 Nama SPPG")
    for i in range(st.session_state.n_sppg):
        v = st.text_input(f"SPPG {i+1}", value=st.session_state.sppg_names[i], key=f"sn_{i}")
        st.session_state.sppg_names[i] = v

    st.markdown("### 🏫 Nama Sekolah")
    for j in range(st.session_state.n_sek):
        v = st.text_input(f"Sekolah {j+1}", value=st.session_state.sek_names[j], key=f"kn_{j}")
        st.session_state.sek_names[j] = v

    st.markdown("---")
    st.markdown('<p style="color:#556B5E;font-size:.7rem;text-align:center">Operasional Riset · Universitas Padjadjaran</p>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-tag">🥗 Program Makan Bergizi Gratis · Kab. Sumedang</div>
  <h1>Optimasi Distribusi MBG</h1>
  <p>Vogel's Approximation Method (VAM) + Modified Distribution Method (MODI) — Solver Transportasi Interaktif</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  INPUT SECTION
# ═══════════════════════════════════════════════
st.markdown('<div class="sec-label">01 · Input Data</div>', unsafe_allow_html=True)
st.markdown('<div class="sec-title">Matriks Biaya, Supply & Demand</div>', unsafe_allow_html=True)

ts = sum(st.session_state.supply); td = sum(st.session_state.demand)
diff = ts - td
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="mcard"><div class="ml">Total Supply</div><div class="mv">{ts:,}</div><div class="ms">porsi</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="mcard"><div class="ml">Total Demand</div><div class="mv">{td:,}</div><div class="ms">siswa</div></div>', unsafe_allow_html=True)
with c3:
    sc = "#1DB87A" if diff==0 else "#F5A623"
    st_txt = "Balanced ✓" if diff==0 else f"Unbalanced ({'+' if diff>0 else ''}{diff:,})"
    note = "siap diproses" if diff==0 else ("dummy demand akan ditambahkan" if diff>0 else "dummy supply akan ditambahkan")
    st.markdown(f'<div class="mcard"><div class="ml">Status</div><div class="mv" style="color:{sc};font-size:1.25rem">{st_txt}</div><div class="ms">{note}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Cost + supply editor
n_r = st.session_state.n_sppg; n_c = st.session_state.n_sek
df_data = {st.session_state.sek_names[j]: [st.session_state.cost[i][j] for i in range(n_r)] for j in range(n_c)}
df_data["📦 Supply"] = st.session_state.supply[:n_r]
df = pd.DataFrame(df_data, index=st.session_state.sppg_names[:n_r])
df.index.name = "SPPG \\ Sekolah"

st.markdown("**✏️ Edit matriks biaya dan supply:**")
edf = st.data_editor(df, use_container_width=True, num_rows="fixed", key="cost_ed")
for i in range(n_r):
    st.session_state.supply[i] = int(edf.iloc[i]["📦 Supply"])
    for j in range(n_c):
        st.session_state.cost[i][j] = int(edf.iloc[i][st.session_state.sek_names[j]])

st.markdown("**✏️ Edit demand:**")
ddf = pd.DataFrame({"Sekolah": st.session_state.sek_names[:n_c], "🎒 Demand (siswa)": st.session_state.demand[:n_c]})
eddf = st.data_editor(ddf, use_container_width=True, num_rows="fixed", key="dem_ed")
for j in range(n_c):
    st.session_state.demand[j] = int(eddf.iloc[j]["🎒 Demand (siswa)"])

st.markdown("<br>", unsafe_allow_html=True)
cb1, cb2 = st.columns([2,1])
with cb1:
    solve_btn = st.button("🚀 Selesaikan dengan VAM + MODI", use_container_width=True)
with cb2:
    if st.button("🔄 Reset ke Data Sumedang", use_container_width=True):
        for k in ["n_sppg","n_sek","sppg_names","sek_names","supply","demand","cost","solved","vam_steps","modi_steps","vam_alloc","modi_alloc","vam_step_idx","modi_step_idx"]:
            if k in st.session_state: del st.session_state[k]
        st.rerun()

if solve_btn:
    c_s = [r[:] for r in st.session_state.cost]
    s_s = st.session_state.supply[:]
    d_s = st.session_state.demand[:]
    rn = st.session_state.sppg_names[:]
    cn = st.session_state.sek_names[:]
    # balance
    ts2 = sum(s_s); td2 = sum(d_s)
    if ts2 > td2:
        d_s.append(ts2-td2); cn.append("Dummy")
        for i in range(len(s_s)): c_s[i].append(0)
    elif td2 > ts2:
        s_s.append(td2-ts2); rn.append("Dummy")
        c_s.append([0]*len(d_s))

    with st.spinner("Menghitung VAM..."):
        va, vs = vam_solve(c_s, s_s, d_s, rn, cn)
    with st.spinner("Mengoptimasi MODI..."):
        ma, ms = modi_solve(c_s, va, rn, cn)

    st.session_state.vam_alloc = va; st.session_state.vam_steps = vs
    st.session_state.modi_alloc = ma; st.session_state.modi_steps = ms
    st.session_state.vam_step_idx = 0; st.session_state.modi_step_idx = 0
    st.session_state.cost_s = c_s; st.session_state.sup_s = s_s
    st.session_state.dem_s = d_s; st.session_state.rnames_s = rn; st.session_state.cnames_s = cn
    st.session_state.solved = True


# ═══════════════════════════════════════════════
#  RESULTS
# ═══════════════════════════════════════════════
if st.session_state.solved:
    vam_steps = st.session_state.vam_steps
    modi_steps = st.session_state.modi_steps
    cost_s = st.session_state.cost_s
    sup_s = st.session_state.sup_s
    dem_s = st.session_state.dem_s
    rn = st.session_state.rnames_s
    cn = st.session_state.cnames_s
    R = len(rn); C = len(cn)

    def tcost(alloc):
        return sum(
            (alloc[i][j] if alloc[i][j] is not None else 0) * cost_s[i][j]
            for i in range(R) for j in range(C)
        )

    vam_final = vam_steps[-1]["alloc"]
    modi_final_raw = st.session_state.modi_alloc
    modi_final = [[v if v is not None else 0 for v in row] for row in modi_final_raw]

    vam_cost = sum((vam_final[i][j] or 0)*cost_s[i][j] for i in range(R) for j in range(C))
    modi_cost = sum((modi_final[i][j] or 0)*cost_s[i][j] for i in range(R) for j in range(C))

    st.markdown("---")

    # ══════════════════════════════
    # PHASE 1: VAM step-by-step
    # ══════════════════════════════
    st.markdown('<div class="sec-label">02 · Fase 1 — VAM</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Vogel\'s Approximation Method — Solusi Awal</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="ibox">
    <b>Cara Kerja VAM:</b> Setiap iterasi, hitung <i>penalty</i> (selisih 2 biaya terkecil) untuk setiap baris dan kolom aktif.
    Pilih baris/kolom dengan penalty tertinggi, lalu alokasikan ke sel berbiaya terendah di baris/kolom tersebut.
    Kolom <b>P1, P2, …</b> di sebelah kanan tabel menampilkan riwayat penalty setiap iterasi.
    </div>
    """, unsafe_allow_html=True)

    n_vam = len(vam_steps)
    vi = st.session_state.vam_step_idx
    step = vam_steps[vi]

    # Step description
    st.markdown(f'<div class="step-desc">{step["desc"]}</div>', unsafe_allow_html=True)

    # Table
    table_html = render_transport_table(
        cost=cost_s,
        alloc=step["alloc"],
        supply=sup_s, demand=dem_s,
        row_names=rn, col_names=cn,
        active_r=step["active_r"], active_c=step["active_c"],
        highlight=step["highlight"],
        history_r=step["history_r"],
        history_c=step["history_c"],
        target_pen=step["target_pen"],
        show_penalties=True,
    )
    st.markdown(table_html, unsafe_allow_html=True)

    # Navigation
    col_prev, col_dots, col_next = st.columns([1, 8, 1])
    with col_prev:
        if st.button("◀", key="vam_prev", disabled=vi==0):
            st.session_state.vam_step_idx -= 1; st.rerun()
    with col_dots:
        dots_html = '<div class="dots">' + ''.join(
            f'<div class="dot{"  active" if i==vi else ""}"></div>' for i in range(n_vam)
        ) + '</div>'
        st.markdown(f'<div style="display:flex;justify-content:center">{dots_html}</div>', unsafe_allow_html=True)
    with col_next:
        if st.button("▶", key="vam_next", disabled=vi==n_vam-1):
            st.session_state.vam_step_idx += 1; st.rerun()

    st.markdown(f"<p style='text-align:center;color:#94a3b8;font-size:.8rem'>Langkah {vi+1} dari {n_vam}</p>", unsafe_allow_html=True)

    # VAM cost
    st.markdown(f"""
    <div style="background:#f0fdf7;border:1.5px solid #1DB87A;border-radius:10px;padding:.9rem 1.3rem;margin:1rem 0">
    💰 <b>Total Biaya Solusi Awal (VAM):</b>
    <span style="font-family:monospace;font-size:1.15rem;color:#0D1F17;font-weight:700">&nbsp;{vam_cost:,.0f}</span> meter·porsi
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ══════════════════════════════
    # PHASE 2: MODI step-by-step
    # ══════════════════════════════
    st.markdown('<div class="sec-label">03 · Fase 2 — MODI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Modified Distribution Method — Optimasi</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="ibox">
    <b>Cara Kerja MODI:</b> (1) Hitung u<sub>i</sub> & v<sub>j</sub> dari sel-sel basic menggunakan C<sub>ij</sub> = u<sub>i</sub> + v<sub>j</sub>.
    (2) Hitung C̄<sub>ij</sub> = C<sub>ij</sub> – u<sub>i</sub> – v<sub>j</sub> untuk sel non-basic — nilai ini muncul <i>di dalam sel kosong tabel</i>.
    (3) Jika ada C̄ &lt; 0, lakukan pergeseran melalui loop. Ulangi sampai semua C̄ ≥ 0.
    Kolom <b>u<sub>i</sub></b> tampil di kiri tabel; baris <b>v<sub>j</sub></b> di bawah.
    </div>
    """, unsafe_allow_html=True)

    n_modi = len(modi_steps)
    mi = st.session_state.modi_step_idx
    mstep = modi_steps[mi]

    # Step description
    st.markdown(f'<div class="step-desc">{mstep["desc"]}</div>', unsafe_allow_html=True)

    # Determine alloc to display (use action_cells for shift phase)
    display_alloc = mstep["alloc"]

    # Table
    m_table_html = render_transport_table(
        cost=cost_s,
        alloc=display_alloc,
        supply=sup_s, demand=dem_s,
        row_names=rn, col_names=cn,
        highlight=mstep.get("highlight"),
        u_vals=mstep.get("u"), v_vals=mstep.get("v"),
        cbar=mstep.get("cbar", {}),
        loop=mstep.get("loop") or [],
        action_cells=mstep.get("action_cells") or {},
        show_penalties=False,
    )
    st.markdown(m_table_html, unsafe_allow_html=True)

    # Navigation
    col_prev2, col_dots2, col_next2 = st.columns([1, 8, 1])
    with col_prev2:
        if st.button("◀", key="modi_prev", disabled=mi==0):
            st.session_state.modi_step_idx -= 1; st.rerun()
    with col_dots2:
        dots2_html = '<div class="dots">' + ''.join(
            f'<div class="dot{"  active" if i==mi else ""}"></div>' for i in range(n_modi)
        ) + '</div>'
        st.markdown(f'<div style="display:flex;justify-content:center">{dots2_html}</div>', unsafe_allow_html=True)
    with col_next2:
        if st.button("▶", key="modi_next", disabled=mi==n_modi-1):
            st.session_state.modi_step_idx += 1; st.rerun()

    st.markdown(f"<p style='text-align:center;color:#94a3b8;font-size:.8rem'>Langkah {mi+1} dari {n_modi}</p>", unsafe_allow_html=True)

    st.markdown("---")

    # ══════════════════════════════
    # FINAL RESULT
    # ══════════════════════════════
    st.markdown('<div class="sec-label">04 · Hasil Akhir</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-title">Tabel Alokasi Optimal</div>', unsafe_allow_html=True)

    final_table = render_transport_table(
        cost=cost_s, alloc=modi_final,
        supply=sup_s, demand=dem_s,
        row_names=rn, col_names=cn,
        show_penalties=False,
    )
    st.markdown(final_table, unsafe_allow_html=True)

    # Route details
    st.markdown("**🗺️ Detail Rute Distribusi Optimal**")
    routes = []
    for i in range(R):
        for j in range(C):
            q = modi_final[i][j]
            if q and q > 0 and "Dummy" not in cn[j]:
                routes.append({"SPPG (Asal)": rn[i], "Sekolah (Tujuan)": cn[j],
                                "Porsi": f"{q:,}", "Jarak (m)": f"{cost_s[i][j]:,}",
                                "Kontribusi Biaya": f"{q*cost_s[i][j]:,}"})
    if routes:
        st.dataframe(pd.DataFrame(routes), use_container_width=True, hide_index=True)

    # Summary
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="mcard"><div class="ml">Biaya Awal (VAM)</div><div class="mv" style="color:#F5A623">{vam_cost:,.0f}</div><div class="ms">meter · porsi</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="mcard"><div class="ml">Biaya Optimal (MODI)</div><div class="mv">{modi_cost:,.0f}</div><div class="ms">meter · porsi</div></div>', unsafe_allow_html=True)
    with m3:
        imp = vam_cost - modi_cost
        pct = imp/vam_cost*100 if vam_cost else 0
        st.markdown(f'<div class="mcard"><div class="ml">Efisiensi</div><div class="mv">{pct:.2f}%</div><div class="ms">Hemat {imp:,.0f} m·porsi</div></div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-banner">
      <h2>🏆 Biaya Distribusi Minimum Optimal</h2>
      <div class="big-cost">{modi_cost:,.0f}</div>
      <div style="opacity:.85;font-size:.95rem;margin-top:.25rem">meter × porsi (total biaya transportasi)</div>
    </div>""", unsafe_allow_html=True)
