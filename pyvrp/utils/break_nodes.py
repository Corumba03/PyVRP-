"""Helpers to add per-vehicle break nodes to a ProblemData instance.

This implements Option B: add one unique break client per vehicle.

    The helper returns a new :class:`~pyvrp._pyvrp.ProblemData` instance
    constructed via :meth:`ProblemData.replace` with extended client lists
    and expanded distance/duration matrices. The break nodes have zero
    travel time to/from all nodes (by default), zero demand and zero
    service duration. By default the break time-window is 11:00--14:00
    (minutes: ``(660, 840)``) unless an explicit ``break_tw`` is given.
"""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from pyvrp._pyvrp import Client, ProblemData
from pyvrp.constants import MAX_VALUE


def add_per_vehicle_break_nodes(
    data: ProblemData,
    num_breaks: int,
    depot_idx: int = 0,
    make_required: bool = True,
    zero_travel: bool = True,
    break_service: int | None = None,
    break_tw: tuple[int, int] | None = None,
) -> Tuple[ProblemData, list[int]]:
    """Return a new ProblemData with ``num_breaks`` break clients appended.

    Parameters
    ----------
    data
        Original problem data instance.
    num_breaks
        Number of break clients to add (typically the fleet size).
    depot_idx
        Index of depot whose coordinates/time-window the break nodes inherit.
    make_required
        If True, the break clients are marked as required (must be visited).
    zero_travel
        If True, travel distance/duration to/from the break nodes is set to
        zero. Otherwise the existing values are copied for new rows/cols.

    Returns
    -------
    new_data, break_indices
        The new :class:`ProblemData` instance and the list of client indices
        (in the full location indexing) of the added break nodes.
    """
    if num_breaks <= 0:
        return data, []

    depots = data.depots()
    if depot_idx < 0 or depot_idx >= len(depots):
        raise IndexError("depot_idx out of range")

    depot = depots[depot_idx]

    # Existing clients (list of Client objects)
    old_clients = data.clients()

    # Create new break Client objects and append to clients list. The client
    # constructor pads load dimensions if necessary.
    break_clients: list[Client] = []
    service = depot.service_duration if break_service is None else break_service
    # Default break time-window: 11:00 to 14:00 in minutes (660..840)
    default_tw = (11 * 60, 14 * 60)
    tw_early = default_tw[0] if break_tw is None else break_tw[0]
    tw_late = default_tw[1] if break_tw is None else break_tw[1]

    for i in range(num_breaks):
        c = Client(
            x=depot.x,
            y=depot.y,
            delivery=[0],
            pickup=[0],
            service_duration=service,
            tw_early=tw_early,
            tw_late=tw_late,
            release_time=0,
            prize=0,
            required=make_required,
            group=None,
            name=f"break_{i}",
        )
        break_clients.append(c)

    new_clients = list(old_clients) + break_clients

    # Expand distance and duration matrices for all profiles.
    old_num_locations = data.num_locations
    new_num_locations = old_num_locations + num_breaks

    new_dist_mats: list[np.ndarray] = []
    new_dur_mats: list[np.ndarray] = []

    for profile in range(data.num_profiles):
        old_dist = data.distance_matrix(profile=profile)
        old_dur = data.duration_matrix(profile=profile)

        # Create new matrices with zeros (or a copy of existing values)
        dist = np.zeros((new_num_locations, new_num_locations), dtype=old_dist.dtype)
        dur = np.zeros((new_num_locations, new_num_locations), dtype=old_dur.dtype)

        # Copy existing submatrix
        dist[:old_num_locations, :old_num_locations] = old_dist
        dur[:old_num_locations, :old_num_locations] = old_dur

        if not zero_travel:
            # If not zero_travel, copy last known values into new rows/cols to
            # keep behaviour similar to existing nodes (rarely desired).
            dist[old_num_locations:, :old_num_locations] = old_dist[0:1, :]
            dist[:old_num_locations, old_num_locations:] = old_dist[:, 0:1]
            dur[old_num_locations:, :old_num_locations] = old_dur[0:1, :]
            dur[:old_num_locations, old_num_locations:] = old_dur[:, 0:1]

        # Ensure diagonals are zero
        np.fill_diagonal(dist, 0)
        np.fill_diagonal(dur, 0)

        new_dist_mats.append(dist)
        new_dur_mats.append(dur)

    # Call replace on ProblemData. Note: replace expects clients list only
    # (depots are unchanged). The matrices must have shape equal to
    # (num_depots + len(clients))**2 which we ensured above.
    new_data = data.replace(
        clients=new_clients,
        distance_matrices=new_dist_mats,
        duration_matrices=new_dur_mats,
    )

    # Break indices in full location indexing: depots first, then clients.
    break_client_start = data.num_locations  # this is old_num_locations
    break_indices = list(range(break_client_start, break_client_start + num_breaks))

    return new_data, break_indices


def add_one_break_per_vehicle(
    data: ProblemData,
    break_service: int,
    break_tw: tuple[int, int],
    depot_idx: int = 0,
) -> Tuple[ProblemData, list[int]]:
    """Return a new ProblemData that enforces one break per vehicle.

    This function appends one break client per vehicle and constructs a
    dedicated routing profile for each vehicle such that only that vehicle
    is allowed to visit its assigned break node. Vehicle types are replaced
    with per-vehicle types (``num_available=1``) that reference the
    corresponding profile. The break nodes are required and inherit the
    given service time and time window.
    """
    num_vehicles = data.num_vehicles
    if num_vehicles <= 0:
        return data, []

    # First create break nodes using the generic helper, supplying the
    # desired service duration and time-window.
    new_data, break_indices = add_per_vehicle_break_nodes(
        data,
        num_breaks=num_vehicles,
        depot_idx=depot_idx,
        make_required=True,
        break_service=break_service,
        break_tw=break_tw,
    )

    old_num_locations = data.num_locations

    # Build new distance/duration matrices: one profile per vehicle. Start
    # from the base matrices of the new data (profile 0) so break nodes
    # (which were added above) are present with their zero-travel values.
    base_dist = new_data.distance_matrix(profile=0)
    base_dur = new_data.duration_matrix(profile=0)

    new_num_locations = new_data.num_locations
    num_depots = new_data.num_depots

    dist_mats: list[np.ndarray] = []
    dur_mats: list[np.ndarray] = []

    # original client indices (full indexing) before breaks: depots..old_num_locations-1
    original_client_idxs = list(range(num_depots, old_num_locations))

    for veh in range(num_vehicles):
        dist = np.array(base_dist, copy=True)
        dur = np.array(base_dur, copy=True)

        # Expand to new size if necessary
        if dist.shape[0] != new_num_locations:
            tmpd = np.full((new_num_locations, new_num_locations), MAX_VALUE, dtype=dist.dtype)
            tmpu = np.full((new_num_locations, new_num_locations), MAX_VALUE, dtype=dur.dtype)
            tmpd[: old_num_locations, : old_num_locations] = dist
            tmpu[: old_num_locations, : old_num_locations] = dur
            dist = tmpd
            dur = tmpu

        # Allowed mask: depots + original clients + assigned break node
        allowed = np.zeros((new_num_locations,), dtype=bool)
        allowed[:num_depots] = True
        allowed[original_client_idxs] = True
        assigned_break_idx = old_num_locations + veh
        allowed[assigned_break_idx] = True

        # For strict one-break-per-vehicle enforcement, forbid other
        # break nodes for this vehicle by setting the corresponding rows
        # and columns to MAX_VALUE. This makes visiting another vehicle's
        # break node impossible for this vehicle.
        for b in range(old_num_locations, new_num_locations):
            if b == assigned_break_idx:
                continue
            dist[b, :] = MAX_VALUE
            dist[:, b] = MAX_VALUE
            dur[b, :] = MAX_VALUE
            dur[:, b] = MAX_VALUE

        # Ensure diagonals are zero after forbids
        np.fill_diagonal(dist, 0)
        np.fill_diagonal(dur, 0)

        dist_mats.append(dist)
        dur_mats.append(dur)

    # Create per-vehicle vehicle types. Use the existing vehicle types as
    # templates and assign them in round-robin order to ensure a stable
    # mapping between vehicle index and profile index.
    orig_vtypes = data.vehicle_types()
    new_vehicle_types = []
    for veh in range(num_vehicles):
        parent = orig_vtypes[veh % len(orig_vtypes)]
        new_vt = parent.replace(num_available=1, profile=veh, name=str(veh))
        new_vehicle_types.append(new_vt)

    # Replace vehicle types and matrices on the new_data object.
    new_data = new_data.replace(
        vehicle_types=new_vehicle_types,
        distance_matrices=dist_mats,
        duration_matrices=dur_mats,
    )

    # Return transformed data and break full-location indices
    break_indices = list(range(old_num_locations, old_num_locations + num_vehicles))
    return new_data, break_indices
