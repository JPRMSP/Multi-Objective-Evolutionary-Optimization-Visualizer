import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random

# ------------------------------
# Benchmark Functions
# ------------------------------
def zdt1(x):
    f1 = x[0]
    g = 1 + 9 * np.mean(x[1:])
    f2 = g * (1 - np.sqrt(f1 / g))
    return [f1, f2]

def zdt2(x):
    f1 = x[0]
    g = 1 + 9 * np.mean(x[1:])
    f2 = g * (1 - (f1 / g) ** 2)
    return [f1, f2]

# ------------------------------
# Utility Functions
# ------------------------------
def dominates(a, b):
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))

def fast_non_dominated_sort(pop_objs):
    fronts = [[]]
    S = [[] for _ in range(len(pop_objs))]
    n = [0] * len(pop_objs)
    rank = [0] * len(pop_objs)

    for p in range(len(pop_objs)):
        for q in range(len(pop_objs)):
            if dominates(pop_objs[p], pop_objs[q]):
                S[p].append(q)
            elif dominates(pop_objs[q], pop_objs[p]):
                n[p] += 1
        if n[p] == 0:
            rank[p] = 0
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1]

# ------------------------------
# Algorithms
# ------------------------------
def nsga2(func, pop_size=50, gens=50, dim=30):
    pop = np.random.rand(pop_size, dim)
    for gen in range(gens):
        objs = [func(ind) for ind in pop]
        fronts = fast_non_dominated_sort(objs)
        new_pop = []
        for front in fronts:
            for idx in front:
                new_pop.append(pop[idx])
            if len(new_pop) >= pop_size:
                break
        pop = np.array(new_pop[:pop_size]) + np.random.normal(0, 0.01, (pop_size, dim))
        pop = np.clip(pop, 0, 1)
    objs = [func(ind) for ind in pop]
    return np.array(objs)

def vega(func, pop_size=50, gens=50, dim=30, objectives=2):
    pop = np.random.rand(pop_size, dim)
    for gen in range(gens):
        new_pop = []
        for i in range(objectives):
            objs = [func(ind)[i] for ind in pop]
            sorted_idx = np.argsort(objs)
            top_half = pop[sorted_idx[:pop_size//2]]
            new_pop.extend(top_half)
        pop = np.array(new_pop[:pop_size]) + np.random.normal(0, 0.01, (pop_size, dim))
        pop = np.clip(pop, 0, 1)
    objs = [func(ind) for ind in pop]
    return np.array(objs)

def spea(func, pop_size=50, gens=50, dim=30):
    pop = np.random.rand(pop_size, dim)
    archive = []
    for gen in range(gens):
        objs = [func(ind) for ind in pop]
        fronts = fast_non_dominated_sort(objs)
        archive = [pop[idx] for idx in fronts[0]]
        if len(archive) > pop_size:
            archive = archive[:pop_size]
        offspring = np.array(archive) + np.random.normal(0, 0.01, (len(archive), dim))
        pop = np.vstack([pop, offspring])
        pop = pop[np.random.choice(len(pop), pop_size, replace=False)]
        pop = np.clip(pop, 0, 1)
    objs = [func(ind) for ind in archive]
    return np.array(objs)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🌟 Multi-Objective Evolutionary Optimization Visualizer")
st.markdown("Compare **NSGA-II, VEGA, SPEA** on ZDT1 and ZDT2 test functions.")

problem = st.selectbox("Select Problem", ["ZDT1", "ZDT2"])
algorithms = st.multiselect("Select Algorithms", ["NSGA-II", "VEGA", "SPEA"], default=["NSGA-II"])
gens = st.slider("Generations", 10, 200, 50)
pop_size = st.slider("Population Size", 20, 200, 50)

if st.button("Run Optimization 🚀"):
    func = zdt1 if problem == "ZDT1" else zdt2
    results_dict = {}

    if "NSGA-II" in algorithms:
        results_dict["NSGA-II"] = nsga2(func, pop_size=pop_size, gens=gens)
    if "VEGA" in algorithms:
        results_dict["VEGA"] = vega(func, pop_size=pop_size, gens=gens)
    if "SPEA" in algorithms:
        results_dict["SPEA"] = spea(func, pop_size=pop_size, gens=gens)

    fig, ax = plt.subplots()
    for algo, results in results_dict.items():
        f1, f2 = results[:,0], results[:,1]
        ax.scatter(f1, f2, label=algo, alpha=0.7)
    ax.set_xlabel("Objective 1")
    ax.set_ylabel("Objective 2")
    ax.set_title(f"Pareto Front Comparison - {problem}")
    ax.legend()
    st.pyplot(fig)
