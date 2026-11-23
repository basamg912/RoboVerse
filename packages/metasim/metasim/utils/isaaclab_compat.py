"""Isaac Lab 2.x / 3.x compatibility helpers shared by the IsaacSim handler and its randomizers.

Isaac Lab 3 (Isaac Sim 6.x) changed the data contract the handler sits on:

* quaternions are (x, y, z, w) instead of (w, x, y, z);
* asset / sensor ``.data.*`` return ``ProxyArray`` instead of ``torch.Tensor``;
* PhysX tensor views (``root_view`` / ``root_physx_view``) take and return ``warp.array``;
* ``isaacsim.core.utils`` (prims, stage) is gone; Isaac Lab owns the stage helpers.

MetaSim's contract stays (w, x, y, z) + ``torch.Tensor``; these helpers convert at the boundary and are
identities on Isaac Lab 2.x. Every import of Isaac Sim / Isaac Lab here is lazy: this module is imported
by code that must load without a running Kit app.
"""

from __future__ import annotations

import functools
import importlib.util

import torch


@functools.cache
def is_isaaclab3() -> bool:
    """Whether the installed Isaac Lab is 3.x. ``isaaclab_physx`` is new in 3.0.

    Checked by module spec rather than distribution version: a source checkout reports the
    ``isaaclab`` extension version (0.45.x for 2.2, 47.x for 3.0), a wheel the release number.
    """
    return importlib.util.find_spec("isaaclab_physx") is not None


def as_torch(value):
    """Isaac Lab 3 wraps asset / sensor data in ``ProxyArray``; 2.x already returns tensors."""
    return getattr(value, "torch", value)


_WXYZ_TO_XYZW = [1, 2, 3, 0]
_XYZW_TO_WXYZ = [3, 0, 1, 2]


def quat_to_sim(quat):
    """MetaSim (w, x, y, z) -> Isaac Lab order. Isaac Lab 3 is (x, y, z, w); 2.x is unchanged.

    Accepts a tensor ``(..., 4)`` or a 4-sequence (config tuples); ``None`` passes through.
    """
    if quat is None or not is_isaaclab3():
        return quat
    if isinstance(quat, torch.Tensor):
        return quat[..., _WXYZ_TO_XYZW]
    w, x, y, z = (float(v) for v in quat)
    return (x, y, z, w)


def quat_from_sim(quat: torch.Tensor) -> torch.Tensor:
    """Isaac Lab quaternion tensor ``(..., 4)`` -> MetaSim (w, x, y, z)."""
    if not is_isaaclab3():
        return quat
    return quat[..., _XYZW_TO_WXYZ]


def pose_to_sim(pose: torch.Tensor) -> torch.Tensor:
    """``(..., 7+)`` MetaSim pose / root state (pos, wxyz, ...) -> Isaac Lab layout."""
    if not is_isaaclab3():
        return pose
    return torch.cat([pose[..., :3], quat_to_sim(pose[..., 3:7]), pose[..., 7:]], dim=-1)


def pose_from_sim(pose: torch.Tensor) -> torch.Tensor:
    """``(..., 7+)`` Isaac Lab pose / root state -> MetaSim (pos, wxyz, ...)."""
    if not is_isaaclab3():
        return pose
    return torch.cat([pose[..., :3], quat_from_sim(pose[..., 3:7]), pose[..., 7:]], dim=-1)


def view_to_torch(value) -> torch.Tensor:
    """A PhysX tensor-view getter result as a torch tensor (``warp.array`` on Isaac Lab 3)."""
    if isinstance(value, torch.Tensor):
        return value
    import warp as wp

    return wp.to_torch(value)


def torch_to_view(value: torch.Tensor):
    """A torch tensor in the form PhysX tensor-view setters take: ``warp.array`` on Isaac Lab 3.

    Values are float32, indices int32 — the dtype follows the tensor, so pass indices as ``torch.int32``.
    """
    if not is_isaaclab3():
        return value
    import warp as wp

    dtype = wp.int32 if value.dtype == torch.int32 else wp.float32
    return wp.from_torch(value.contiguous(), dtype=dtype)


def view_indices(env_ids: torch.Tensor):
    """Env indices for a PhysX tensor-view setter: CPU ``int32`` ``warp.array`` on Isaac Lab 3, unchanged on 2.x."""
    if not is_isaaclab3():
        return env_ids
    return torch_to_view(env_ids.to(device="cpu", dtype=torch.int32))


def physx_view(asset):
    """The PhysX tensor view of an asset: ``root_view`` on Isaac Lab 3 (``root_physx_view`` is deprecated)."""
    view = getattr(asset, "root_view", None)
    return view if view is not None else asset.root_physx_view


def get_current_stage():
    """The open USD stage. Isaac Lab 3 owns the helper; Isaac Sim 6 dropped ``isaacsim.core.utils``."""
    try:
        from isaaclab.sim.utils.stage import get_current_stage
    except ImportError:
        try:
            from omni.isaac.core.utils.prims import get_current_stage
        except ImportError:
            from isaacsim.core.utils.prims import get_current_stage
    return get_current_stage()


def add_usd_reference(usd_path: str, prim_path: str):
    """Reference ``usd_path`` at ``prim_path`` (an Xform is created if missing) and return the prim."""
    try:
        from isaaclab.sim.utils.prims import add_usd_reference as _add

        return _add(prim_path, usd_path)
    except ImportError:
        pass
    try:
        from isaacsim.core.utils.stage import add_reference_to_stage
    except ImportError:
        try:
            from omni.isaac.core.utils.stage import add_reference_to_stage
        except ImportError:
            add_reference_to_stage = None
    if add_reference_to_stage is not None:
        return add_reference_to_stage(usd_path, prim_path)
    stage = get_current_stage()
    prim = stage.DefinePrim(prim_path, "Xform")
    prim.GetReferences().AddReference(usd_path)
    return prim


# ---------------------------------------------------------------------------------------------------------------------
# Per-body / per-shape physics properties. All tensors are CPU (the PhysX tensor API keeps these on the host) and
# indexed in the asset's public ``body_names`` order. Isaac Lab 3 writes go through the asset so its cached
# ``data.body_mass`` / ``data.body_inertia`` stay coherent with the simulation.
# ---------------------------------------------------------------------------------------------------------------------


def get_body_masses(asset) -> torch.Tensor:
    """Body masses, ``(num_envs, num_bodies)`` for an articulation, ``(num_envs, 1)`` for a rigid object."""
    if is_isaaclab3():
        return as_torch(asset.data.body_mass).detach().to("cpu").clone()
    return asset.root_physx_view.get_masses()


def set_body_masses(asset, masses: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write the ``env_ids`` rows of a full ``(num_envs, num_bodies)`` mass tensor."""
    if is_isaaclab3():
        asset.set_masses_index(masses=masses[env_ids].to(asset.device), env_ids=env_ids.to(asset.device))
    else:
        asset.root_physx_view.set_masses(masses, env_ids)


def get_body_inertias(asset) -> torch.Tensor:
    """Body inertias (row-major 3x3), ``(num_envs, num_bodies, 9)`` / ``(num_envs, 9)``."""
    if is_isaaclab3():
        return as_torch(asset.data.body_inertia).detach().to("cpu").clone()
    return asset.root_physx_view.get_inertias()


def get_default_inertias(asset) -> torch.Tensor:
    """Default body inertias, same layout as :func:`get_body_inertias`, on CPU."""
    return as_torch(asset.data.default_inertia).detach().to("cpu")


def set_body_inertias(asset, inertias: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write the ``env_ids`` rows of a full inertia tensor."""
    if is_isaaclab3():
        asset.set_inertias_index(inertias=inertias[env_ids].to(asset.device), env_ids=env_ids.to(asset.device))
    else:
        asset.root_physx_view.set_inertias(inertias, env_ids)


def get_shape_materials(asset) -> torch.Tensor:
    """Per-shape (static friction, dynamic friction, restitution), ``(num_envs, num_shapes, 3)``, CPU."""
    return view_to_torch(physx_view(asset).get_material_properties())


def set_shape_materials(asset, materials: torch.Tensor, env_ids: torch.Tensor) -> None:
    """Write the ``env_ids`` rows of a full per-shape material tensor."""
    physx_view(asset).set_material_properties(torch_to_view(materials), view_indices(env_ids))


def num_shapes_per_body(asset) -> tuple[list[int], int]:
    """Collision-shape count of each articulation link (PhysX link order) and the asset's total shape count."""
    view = physx_view(asset)
    counts = [asset._physics_sim_view.create_rigid_body_view(path).max_shapes for path in view.link_paths[0]]
    return counts, view.max_shapes


def body_ids_to_physx(asset, body_ids):
    """Public ``body_names`` indices -> PhysX link order. Identity unless Isaac Lab 3 ``body_ordering`` is set."""
    mapper = getattr(asset, "map_body_ids_to_backend", None)
    return mapper(body_ids) if callable(mapper) else body_ids
