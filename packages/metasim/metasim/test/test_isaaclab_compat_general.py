"""The Isaac Lab 2.x / 3.x compat helpers keep MetaSim's (w, x, y, z) + ``torch.Tensor`` contract.

Isaac Lab 3 switched quaternions to (x, y, z, w) and wraps asset data in ``ProxyArray``; the helpers convert at
the handler boundary and must be identities on Isaac Lab 2.x. Exercised without Isaac Lab installed by forcing
``is_isaaclab3`` either way.
"""

from __future__ import annotations

import pytest
import torch

from metasim.utils import isaaclab_compat as compat


@pytest.fixture
def isaaclab3(monkeypatch):
    monkeypatch.setattr(compat, "is_isaaclab3", lambda: True)


@pytest.fixture
def isaaclab2(monkeypatch):
    monkeypatch.setattr(compat, "is_isaaclab3", lambda: False)


@pytest.mark.general
def test_isaaclab2_conversions_are_identities(isaaclab2):
    quat = torch.tensor([[0.9, 0.1, 0.2, 0.3]])
    state = torch.arange(13.0).reshape(1, 13)
    assert compat.quat_to_sim(quat) is quat
    assert compat.quat_from_sim(quat) is quat
    assert compat.quat_to_sim((1.0, 0.0, 0.0, 0.0)) == (1.0, 0.0, 0.0, 0.0)
    assert compat.pose_to_sim(state) is state
    assert compat.pose_from_sim(state) is state
    ids = torch.tensor([0, 2])
    assert compat.view_indices(ids) is ids
    assert compat.torch_to_view(state) is state


@pytest.mark.general
def test_isaaclab3_quaternions_are_xyzw(isaaclab3):
    assert compat.quat_to_sim((1.0, 0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0, 1.0)
    assert compat.quat_to_sim(None) is None
    wxyz = torch.tensor([[0.9, 0.1, 0.2, 0.3], [0.5, -0.5, 0.5, -0.5]])
    xyzw = compat.quat_to_sim(wxyz)
    torch.testing.assert_close(xyzw, torch.tensor([[0.1, 0.2, 0.3, 0.9], [-0.5, 0.5, -0.5, 0.5]]))
    torch.testing.assert_close(compat.quat_from_sim(xyzw), wxyz)


@pytest.mark.general
def test_isaaclab3_pose_round_trip_keeps_position_and_velocity(isaaclab3):
    state = torch.randn(4, 3, 13)  # (envs, bodies, pos + quat + lin vel + ang vel)
    sim = compat.pose_to_sim(state)
    torch.testing.assert_close(sim[..., :3], state[..., :3])
    torch.testing.assert_close(sim[..., 3:7], state[..., [4, 5, 6, 3]])
    torch.testing.assert_close(sim[..., 7:], state[..., 7:])
    torch.testing.assert_close(compat.pose_from_sim(sim), state)
    # 7-dim poses (write_root_pose_to_sim) take the same path
    torch.testing.assert_close(compat.pose_from_sim(compat.pose_to_sim(state[..., :7])), state[..., :7])


@pytest.mark.general
def test_world_state_to_env_local_returns_wxyz(isaaclab3):
    from metasim.sim.isaacsim.isaacsim import _world_state_to_env_local

    origins = torch.tensor([[10.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    state_w = torch.zeros(2, 13)
    state_w[:, :3] = origins + 1.0
    state_w[:, 3:7] = torch.tensor([0.1, 0.2, 0.3, 0.9])  # Isaac Lab 3 (x, y, z, w)
    local = _world_state_to_env_local(_ProxyArrayLike(state_w), origins)
    torch.testing.assert_close(local[:, :3], torch.ones(2, 3))
    torch.testing.assert_close(local[:, 3:7], torch.tensor([0.9, 0.1, 0.2, 0.3]).expand(2, 4))
    assert state_w[0, 3] == pytest.approx(0.1), "the source buffer must not be modified"


class _ProxyArrayLike:
    """Stand-in for Isaac Lab 3's ``ProxyArray``: exposes the tensor as ``.torch``."""

    def __init__(self, tensor: torch.Tensor):
        self.torch = tensor


@pytest.mark.general
def test_as_torch_unwraps_proxy_arrays_and_passes_tensors():
    tensor = torch.ones(2)
    assert compat.as_torch(tensor) is tensor
    assert compat.as_torch(_ProxyArrayLike(tensor)) is tensor
    assert compat.as_torch(None) is None


@pytest.mark.general
def test_body_ids_map_to_physx_order_only_when_the_asset_reorders():
    class Plain:
        pass

    class Reordered:
        def map_body_ids_to_backend(self, ids):
            return [i + 10 for i in ids]

    assert compat.body_ids_to_physx(Plain(), [0, 1]) == [0, 1]
    assert compat.body_ids_to_physx(Reordered(), [0, 1]) == [10, 11]
