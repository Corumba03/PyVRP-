"""Utilities to insert per-vehicle break nodes into routes (prototype).

This module provides a small, Python-side greedy repair used by the
notebook/experiments. It is intentionally simple: for each route in a
`pyvrp._pyvrp.Solution` we assign one of the break nodes (one per
vehicle) and insert it near the start of the route. The purpose is to
produce a displayable set of routes that each contain exactly one
break node so we can visualise and sanity-check break handling.

This is a prototype helper only — a production implementation should
validate feasibility properly and be integrated into the decoder.
"""

from typing import List, Tuple


def repair_one_break_per_route(data, solution, break_indices: List[int]) -> Tuple[List[List[int]], bool]:
    """Produce final routes with exactly one break node per route.

    Parameters
    - data: a `pyvrp.ProblemData` instance (not used heavily here,
      present to match the notebook API)
    - solution: a `pyvrp.Solution` object (the solver result)
    - break_indices: list of full-index node ids representing break
      clients (one per vehicle)

    Returns a tuple `(final_routes, all_ok)` where `final_routes` is a
    list of routes (each route is a list of full-index node ids) and
    `all_ok` is True when insertion succeeded for all routes.

    Notes
    - This function assigns break nodes in order to routes in the
      solution. It inserts the break as the first client visit (after
      the depot) for readability and plotting.
    - The function does not perform a full feasibility check; it is a
      lightweight helper for prototyping and visual inspection.
    """
    final_routes: List[List[int]] = []
    used_breaks = set()

    # `solution.routes()` returns a list of Route objects; call
    # .visits() on each route to get the full-index visit list.
    try:
        routes = solution.routes()
    except Exception:
        # Fallback: if solution is a plain list of routes already
        routes = solution

    bi_iter = iter(break_indices)

    all_ok = True
    for r in routes:
        try:
            visits = list(r.visits())
        except Exception:
            # If r is already a list of ints
            visits = list(r)

        # Assign the next unused break index if available
        assigned_break = None
        for b in break_indices:
            if b not in used_breaks:
                assigned_break = b
                used_breaks.add(b)
                break

        if assigned_break is None:
            # No break available — mark failure but keep original route
            all_ok = False
            final_routes.append(visits)
            continue

        # Insert break as the first client visit: after the starting depot
        insert_pos = 1 if len(visits) >= 1 else 0
        visits.insert(insert_pos, assigned_break)
        final_routes.append(visits)

    return final_routes, all_ok


def remove_break_only_routes(final_routes: List[List[int]], break_indices: List[int]) -> Tuple[List[List[int]], int]:
    """Filter out routes that contain only a break (and depots).

    Parameters
    - final_routes: list of routes (full-index node ids)
    - break_indices: list of break node ids

    Returns `(filtered_routes, removed_count)` where `filtered_routes`
    contains only routes that have at least one non-break client.
    """
    filtered = []
    removed = 0

    # A visit is a client if it is not in the break_indices and is
    # not an obvious depot (depots are usually small indices; we treat
    # any node that is not a break as a client for this filter).
    break_set = set(break_indices)

    for r in final_routes:
        has_non_break = any((v not in break_set) for v in r)
        # Also ensure the route contains some client different from a depot
        # (we keep it conservative: if every visit is a break we remove it)
        if not has_non_break:
            removed += 1
        else:
            filtered.append(r)

    return filtered, removed
