"""
Pyomo + IPOPT budget sweep for €/TRUE DIAGNOSED case, with:
- Higher PPV constraint (default PPV >= 0.80)
- Progress bar
- Full listing table
- Plot opens in a window when possible (interactive backend); otherwise saves fallback PNG

Model:
- Binormal equal-variance score per block (AUC->delta via scipy)
- Top-q selection of the MIXTURE via threshold variables t_k
- Uses logistic approximation to Normal CDF inside Pyomo (no pyo.erf dependency)

If you want exact Normal CDF, use the SciPy-evaluation grid/sweep code instead of Pyomo.
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

import pyomo.environ as pyo
from scipy.stats import norm

# progress bar (optional)
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:
    tqdm = None

# tables (optional)
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None


# ----------------------------- Inputs -----------------------------

@dataclass
class Block:
    name: str
    auc: float
    cost: float


def delta_from_auc(auc: float) -> float:
    return math.sqrt(2.0) * float(norm.ppf(auc))


# Logistic approximation to standard normal CDF (smooth, Pyomo-friendly)
A_PHI = 1.702
def Phi(x):
    return 1.0 / (1.0 + pyo.exp(-A_PHI * x))


# ----------------------------- Pyomo Model -----------------------------

def build_model(
    blocks: List[Block],
    pi0: float = 0.01,
    budget_case_init: float = 10_000.0,
    ppv_min: float = 0.80,                  # <-- stricter default
    max_imaging_frac: float = 1.0,          # non-binding by default
    max_final_positive: float = 1.0,        # non-binding by default
) -> pyo.ConcreteModel:
    deltas = [delta_from_auc(b.auc) for b in blocks]
    costs = [b.cost for b in blocks]
    K = len(blocks)

    m = pyo.ConcreteModel()
    m.K = pyo.RangeSet(1, K)
    m._K = K
    m._block_names = [b.name for b in blocks]
    m._pi0 = float(pi0)

    # Mutable parameters
    m.BUDGET_CASE = pyo.Param(initialize=float(budget_case_init), mutable=True)
    m.PPV_MIN = pyo.Param(initialize=float(ppv_min), mutable=True)
    m.MAX_IMAGING = pyo.Param(initialize=float(max_imaging_frac), mutable=True)
    m.MAX_FINALPOS = pyo.Param(initialize=float(max_final_positive), mutable=True)

    eps = 1e-4
    m.q = pyo.Var(m.K, bounds=(eps, 1.0 - eps))    # sliders
    m.t = pyo.Var(m.K, bounds=(-12.0, 12.0))       # thresholds

    m.pi = pyo.Var(pyo.RangeSet(0, K), bounds=(eps, 1.0 - eps))
    m.r  = pyo.Var(pyo.RangeSet(0, K), bounds=(eps, 1.0))
    m.S  = pyo.Var(pyo.RangeSet(0, K), bounds=(eps, 1.0))

    # init
    m.pi[0].fix(float(pi0))
    m.r[0].fix(1.0)
    m.S[0].fix(1.0)

    def TPR(m, k):
        return 1.0 - Phi(m.t[k] - deltas[k - 1])

    def FPR(m, k):
        return 1.0 - Phi(m.t[k])

    def mix_pass(m, k):
        return (1.0 - m.pi[k - 1]) * FPR(m, k) + m.pi[k - 1] * TPR(m, k)

    # mixture top-q
    m.mix_def = pyo.Constraint(m.K, rule=lambda m, k: mix_pass(m, k) == m.q[k])

    # propagation
    m.r_update  = pyo.Constraint(m.K, rule=lambda m, k: m.r[k] == m.r[k - 1] * m.q[k])
    m.S_update  = pyo.Constraint(m.K, rule=lambda m, k: m.S[k] == m.S[k - 1] * TPR(m, k))
    m.pi_update = pyo.Constraint(m.K, rule=lambda m, k: m.pi[k] * m.q[k] == m.pi[k - 1] * TPR(m, k))

    # costs
    m.cost_per_person = pyo.Expression(expr=sum(costs[k - 1] * m.r[k - 1] for k in m.K))
    m.diagnosed_per_person = pyo.Expression(expr=float(pi0) * m.S[K])
    m.cost_per_case = pyo.Expression(expr=m.cost_per_person / m.diagnosed_per_person)

    # constraints
    m.budget_case = pyo.Constraint(expr=m.cost_per_case <= m.BUDGET_CASE)
    m.ppv_constraint = pyo.Constraint(expr=m.pi[K] >= m.PPV_MIN)
    m.imaging_cap = pyo.Constraint(expr=m.r[K - 1] <= m.MAX_IMAGING)
    m.finalpos_cap = pyo.Constraint(expr=m.r[K] <= m.MAX_FINALPOS)

    # objective: maximize sensitivity
    m.obj = pyo.Objective(expr=-m.S[K], sense=pyo.minimize)
    return m


# ----------------------------- IPOPT helpers -----------------------------

def seed_initial_values(m: pyo.ConcreteModel, rng: random.Random):
    K = m._K
    for k in m.K:
        if k <= 2:
            q0 = rng.uniform(0.60, 0.98)
        elif k == 3:
            q0 = rng.uniform(0.03, 0.30)
        else:
            q0 = rng.uniform(0.05, 0.95)
        m.q[k].set_value(q0)

        t0 = float(norm.ppf(max(1e-6, min(1 - 1e-6, 1 - q0))))
        m.t[k].set_value(max(-10.0, min(10.0, t0)))

    m.r[0].set_value(1.0)
    m.S[0].set_value(1.0)
    for k in range(1, K + 1):
        m.r[k].set_value(max(1e-4, float(pyo.value(m.r[k - 1])) * float(pyo.value(m.q[k]))))
        m.S[k].set_value(max(1e-4, float(pyo.value(m.S[k - 1])) * 0.6))
        m.pi[k].set_value(max(1e-4, min(1 - 1e-4, float(pyo.value(m.pi[k - 1])) / max(1e-4, float(pyo.value(m.q[k]))))))


def extract_solution(m: pyo.ConcreteModel) -> Dict[str, Any]:
    K = m._K
    q = [float(pyo.value(m.q[k])) for k in m.K]
    return {
        "q": q,
        "sensitivity": float(pyo.value(m.S[K])),
        "final_ppv": float(pyo.value(m.pi[K])),
        "cost_per_person": float(pyo.value(m.cost_per_person)),
        "cost_per_case": float(pyo.value(m.cost_per_case)),
        "reach_imaging": float(pyo.value(m.r[K - 1])),
        "final_positive": float(pyo.value(m.r[K])),
        "objective": float(pyo.value(m.obj)),
    }


def solve_multistart(
    m: pyo.ConcreteModel,
    n_starts: int = 30,
    seed: int = 1,
    tee: bool = False,
) -> Dict[str, Any]:
    solver = pyo.SolverFactory("ipopt")
    solver.options["max_iter"] = 4000
    solver.options["tol"] = 1e-8
    solver.options["print_level"] = 5 if tee else 0

    rng = random.Random(seed)
    best = None
    best_obj = float("inf")

    for _ in range(n_starts):
        seed_initial_values(m, rng)
        try:
            res = solver.solve(m, tee=tee)
        except Exception:
            continue

        term = str(res.solver.termination_condition).lower()
        if not any(x in term for x in ["optimal", "locallyoptimal", "feasible"]):
            continue

        sol = extract_solution(m)
        if sol["objective"] < best_obj:
            best_obj = sol["objective"]
            best = sol

    if best is None:
        raise RuntimeError(
            "No feasible solution found. Common reasons:\n"
            "- PPV_MIN too strict for given AUCs/costs (try 0.8 -> 0.7, or relax imaging cap)\n"
            "- budget range too low\n"
            "- need more starts"
        )
    return best


# ----------------------------- Sweet spot -----------------------------

def knee_point(costs: List[float], sens: List[float]) -> int:
    import numpy as np
    x = np.array(costs, dtype=float)
    y = np.array(sens, dtype=float)
    if len(x) < 3:
        return 0
    x_n = (x - x.min()) / (x.max() - x.min() + 1e-12)
    y_n = (y - y.min()) / (y.max() - y.min() + 1e-12)
    x1, y1 = x_n[0], y_n[0]
    x2, y2 = x_n[-1], y_n[-1]
    A = y1 - y2
    B = x2 - x1
    C = x1 * y2 - x2 * y1
    dist = abs(A * x_n + B * y_n + C) / (math.sqrt(A*A + B*B) + 1e-12)
    return int(dist.argmax())


def make_budgets_1000_10000() -> List[float]:
    budgets = []
    budgets += list(range(1000, 3001, 100))
    budgets += list(range(3000, 8001, 250))
    budgets += list(range(8000, 10001, 200))
    return [float(b) for b in budgets]


def sweep(
    blocks: List[Block],
    pi0: float,
    budgets: List[float],
    ppv_min: float = 0.80,
    max_imaging_frac: float = 1.0,
    max_final_positive: float = 1.0,
    n_starts_per_budget: int = 25,
    seed: int = 1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    m = build_model(
        blocks=blocks,
        pi0=pi0,
        budget_case_init=budgets[0],
        ppv_min=ppv_min,
        max_imaging_frac=max_imaging_frac,
        max_final_positive=max_final_positive,
    )

    out: List[Dict[str, Any]] = []
    it = budgets
    if tqdm is not None:
        it = tqdm(budgets, desc="Sweep €/diagnosed", leave=True)

    for b in it:
        m.BUDGET_CASE.set_value(float(b))
        m.PPV_MIN.set_value(float(ppv_min))
        m.MAX_IMAGING.set_value(float(max_imaging_frac))
        m.MAX_FINALPOS.set_value(float(max_final_positive))

        sol = solve_multistart(m, n_starts=n_starts_per_budget, seed=seed, tee=False)
        rec = {
            "budget_case": float(b),
            "cost_per_case": sol["cost_per_case"],
            "cost_per_person": sol["cost_per_person"],
            "sensitivity": sol["sensitivity"],
            "final_ppv": sol["final_ppv"],
            "reach_imaging": sol["reach_imaging"],
            "final_positive": sol["final_positive"],
        }
        for name, qk in zip([blk.name for blk in blocks], sol["q"]):
            rec[f"q_{name}"] = qk
        out.append(rec)

    costs = [r["cost_per_case"] for r in out]
    sens = [r["sensitivity"] for r in out]
    k = knee_point(costs, sens)
    return out, out[k]


# ----------------------------- Plot (interactive window if possible) -----------------------------

def plot_curve_interactive(results: List[Dict[str, Any]], sweet: Dict[str, Any], fallback_png: str = "sweetspot_curve.png"):
    # Try to force an interactive backend BEFORE importing pyplot.
    # If this fails, we fall back to saving a PNG.
    try:
        import matplotlib
        # Try a couple of common interactive backends
        for backend in ["QtAgg", "TkAgg", "MacOSX"]:
            try:
                matplotlib.use(backend, force=True)
                break
            except Exception:
                continue
        import matplotlib.pyplot as plt  # type: ignore

        xs = [r["cost_per_case"] for r in results]
        ys = [r["sensitivity"] * 100 for r in results]

        plt.figure()
        plt.plot(xs, ys, marker=".")
        plt.scatter([sweet["cost_per_case"]], [sweet["sensitivity"] * 100], marker="o")
        plt.xlabel("€ per true diagnosed case")
        plt.ylabel("Sensitivity (%)")
        plt.title("Sensitivity vs €/diagnosed (sweet spot marked)")
        plt.grid(True, alpha=0.3)

        # Force window display
        plt.show(block=True)
    except Exception as e:
        # fallback to file
        try:
            import matplotlib.pyplot as plt  # type: ignore
            plt.figure()
            xs = [r["cost_per_case"] for r in results]
            ys = [r["sensitivity"] * 100 for r in results]
            plt.plot(xs, ys, marker=".")
            plt.scatter([sweet["cost_per_case"]], [sweet["sensitivity"] * 100], marker="o")
            plt.xlabel("€ per true diagnosed case")
            plt.ylabel("Sensitivity (%)")
            plt.title("Sensitivity vs €/diagnosed (sweet spot marked)")
            plt.grid(True, alpha=0.3)
            plt.savefig(fallback_png, dpi=200)
            print(f"\nCould not open interactive plot window ({e!r}). Saved: {fallback_png}")
        except Exception as e2:
            print(f"\nCould not plot at all: {e2!r}")


# ----------------------------- Main -----------------------------

if __name__ == "__main__":
    # Your stated defaults (edit as needed)
    blocks = [
        Block("PRS",     auc=0.75, cost=0.0),
        Block("Blood",   auc=0.75, cost=0.0),
        Block("Fecal",   auc=0.85, cost=25.0),
        Block("Omics",   auc=0.85, cost=100.0),
        Block("Imaging", auc=0.90, cost=200.0),
    ]

    pi0 = 0.01
    budgets = make_budgets_1000_10000()

    # Make “diagnosed” actually diagnosis-grade:
    PPV_MIN = 0.80          # bump to 0.90 if you want stricter diagnosis
    # Optional operational caps:
    MAX_IMAGING = 0.05      # e.g. <=5% sent to imaging (set 1.0 to disable)
    MAX_FINALPOS = 0.02     # e.g. <=2% end positive (set 1.0 to disable)

    results, sweet = sweep(
        blocks=blocks,
        pi0=pi0,
        budgets=budgets,
        ppv_min=PPV_MIN,
        max_imaging_frac=MAX_IMAGING,
        max_final_positive=MAX_FINALPOS,
        n_starts_per_budget=30,
        seed=1,
    )

    print("\nSWEET SPOT")
    print(f"  €/diagnosed:   €{sweet['cost_per_case']:,.0f}")
    print(f"  Sensitivity:   {sweet['sensitivity']*100:,.2f}%")
    print(f"  Final PPV:     {sweet['final_ppv']*100:,.2f}%")
    print(f"  € per person:  €{sweet['cost_per_person']:,.2f}")
    print("  Sliders:")
    for b in blocks:
        print(f"    {b.name:10s}: {sweet[f'q_{b.name}']*100:6.2f}%")

    # Full listing
    if pd is not None:
        df = pd.DataFrame(results).copy()
        for b in blocks:
            df[f"q_{b.name}"] = (df[f"q_{b.name}"] * 100).round(2)
        df["sensitivity_%"] = (df["sensitivity"] * 100).round(2)
        df["final_ppv_%"] = (df["final_ppv"] * 100).round(2)
        df["reach_imaging_%"] = (df["reach_imaging"] * 100).round(3)
        df["final_positive_%"] = (df["final_positive"] * 100).round(3)
        df["cost_per_case"] = df["cost_per_case"].round(0).astype(int)
        df["cost_per_person"] = df["cost_per_person"].round(2)

        cols = ["budget_case","cost_per_case","cost_per_person","sensitivity_%","final_ppv_%","reach_imaging_%","final_positive_%"] + \
               [f"q_{b.name}" for b in blocks]
        print("\nFULL LISTING (all rows):")
        print(df[cols].to_string(index=False))

    plot_curve_interactive(results, sweet)
