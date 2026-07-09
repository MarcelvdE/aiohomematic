# SPDX-License-Identifier: MIT
# Copyright (c) 2021-2026
"""
Tests for DeviceDetailsCache freshness, persistence, and CentralDataCache loading.

These tests cover the CCU-load reductions around device metadata:
- refresh() is guarded by DEVICE_DETAILS_MAX_CACHE_AGE instead of being always-stale
- the details cache persists to disk and is reusable across restarts
- CentralDataCache.load() skips only fresh interfaces instead of aborting the loop
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from aiohomematic.async_support import Looper
from aiohomematic.const import DataOperationResult, Interface
from aiohomematic.store import LocalStorageFactory
from aiohomematic.store.dynamic import CentralDataCache, DeviceDetailsCache

# pylint: disable=protected-access


def _make_client() -> MagicMock:
    """Create a mock primary client whose fetch populates names like the real one."""
    client = MagicMock()
    client.fetch_device_details = AsyncMock()
    client.get_all_rooms = AsyncMock(return_value={"VCU0000001:1": {"Living Room"}})
    client.get_all_functions = AsyncMock(return_value={"VCU0000001:1": {"Light"}})
    return client


def _make_details_cache(*, tmp_path, client: MagicMock | None) -> DeviceDetailsCache:
    """Create a DeviceDetailsCache backed by real local storage."""
    central_info = MagicMock()
    central_info.name = "test_central"
    config_provider = MagicMock()
    config_provider.config.use_caches = True
    primary_client_provider = MagicMock()
    primary_client_provider.primary_client = client
    storage_factory = LocalStorageFactory(
        base_directory=str(tmp_path),
        central_name="test_central",
        task_scheduler=Looper(),
    )
    storage = storage_factory.create_storage(key="homematic_device_details", sub_directory="cache")
    return DeviceDetailsCache(
        central_info=central_info,
        config_provider=config_provider,
        primary_client_provider=primary_client_provider,
        storage=storage,
    )


class TestDeviceDetailsRefresh:
    """Test refresh() freshness guard and client handling."""

    @pytest.mark.asyncio
    async def test_refresh_direct_call_bypasses_ttl(self, tmp_path) -> None:
        """direct_call=True must force a fetch even while the cache is fresh."""
        client = _make_client()
        cache = _make_details_cache(tmp_path=tmp_path, client=client)

        await cache.refresh()
        await cache.refresh(direct_call=True)

        assert client.fetch_device_details.await_count == 2

    @pytest.mark.asyncio
    async def test_refresh_is_guarded_by_long_ttl(self, tmp_path) -> None:
        """A second refresh within DEVICE_DETAILS_MAX_CACHE_AGE must not re-fetch."""
        client = _make_client()
        cache = _make_details_cache(tmp_path=tmp_path, client=client)

        await cache.refresh()
        await cache.refresh()

        # Only the first call may hit the backend (Device.listAllDetail etc.)
        assert client.fetch_device_details.await_count == 1
        assert client.get_all_rooms.await_count == 1
        assert client.get_all_functions.await_count == 1

    @pytest.mark.asyncio
    async def test_refresh_populates_rooms_and_functions(self, tmp_path) -> None:
        """refresh() must populate channel/device rooms and functions."""
        client = _make_client()
        cache = _make_details_cache(tmp_path=tmp_path, client=client)

        await cache.refresh()

        assert cache.get_channel_rooms(channel_address="VCU0000001:1") == {"Living Room"}
        assert cache.get_device_rooms(device_address="VCU0000001") == {"Living Room"}
        assert cache.get_function_text(address="VCU0000001:1") == "Light"

    @pytest.mark.asyncio
    async def test_refresh_without_client_keeps_cache_and_stays_stale(self, tmp_path) -> None:
        """Without a primary client, refresh() must not clear data or mark itself fresh."""
        cache = _make_details_cache(tmp_path=tmp_path, client=None)
        cache.add_name(address="VCU0000001", name="Device")

        await cache.refresh()

        assert cache.get_name(address="VCU0000001") == "Device"

        # A client appearing later must lead to an actual fetch (not a fresh no-op).
        client = _make_client()
        cache._primary_client_provider.primary_client = client
        await cache.refresh()
        assert client.fetch_device_details.await_count == 1


class TestDeviceDetailsPersistence:
    """Test disk persistence of the device details cache."""

    @pytest.mark.asyncio
    async def test_clear_in_memory_keeps_ise_ids_and_interfaces(self, tmp_path) -> None:
        """clear_in_memory() must keep ise_ids/interfaces (pre-existing semantics)."""
        cache = _make_details_cache(tmp_path=tmp_path, client=None)
        cache.add_name(address="VCU0000001", name="Device")
        cache.add_address_ise_id(address="VCU0000001", ise_id=4711)
        cache.add_interface(address="VCU0000001", interface=Interface.HMIP_RF)

        cache.clear_in_memory()

        assert cache.get_name(address="VCU0000001") is None
        assert cache.get_address_id(address="VCU0000001") == 4711
        assert cache.get_interface(address="VCU0000001") == Interface.HMIP_RF

    @pytest.mark.asyncio
    async def test_clear_removes_persisted_content(self, tmp_path) -> None:
        """clear() must remove persisted content so a reload yields nothing."""
        cache = _make_details_cache(tmp_path=tmp_path, client=_make_client())
        cache.add_name(address="VCU0000001", name="Device")
        await cache.save()

        await cache.clear()

        restored = _make_details_cache(tmp_path=tmp_path, client=None)
        assert await restored.load() != DataOperationResult.LOAD_SUCCESS
        assert restored.get_name(address="VCU0000001") is None

    @pytest.mark.asyncio
    async def test_loaded_content_is_not_considered_fresh(self, tmp_path) -> None:
        """After a disk load, the next refresh() must still fetch from the backend."""
        cache = _make_details_cache(tmp_path=tmp_path, client=_make_client())
        cache.add_name(address="VCU0000001", name="Device")
        await cache.save()

        client = _make_client()
        restored = _make_details_cache(tmp_path=tmp_path, client=client)
        assert await restored.load() == DataOperationResult.LOAD_SUCCESS

        await restored.refresh()
        assert client.fetch_device_details.await_count == 1

    @pytest.mark.asyncio
    async def test_save_and_load_roundtrip(self, tmp_path) -> None:
        """Persisted details must be restored by a fresh cache instance."""
        client = _make_client()
        cache = _make_details_cache(tmp_path=tmp_path, client=client)
        cache.add_name(address="VCU0000001", name="Device")
        cache.add_name(address="VCU0000001:1", name="Device:1")
        cache.add_address_ise_id(address="VCU0000001", ise_id=4711)
        cache.add_interface(address="VCU0000001", interface=Interface.HMIP_RF)
        await cache.refresh()

        restored = _make_details_cache(tmp_path=tmp_path, client=None)
        assert await restored.load() == DataOperationResult.LOAD_SUCCESS
        assert restored.get_name(address="VCU0000001") == "Device"
        assert restored.get_name(address="VCU0000001:1") == "Device:1"
        assert restored.get_address_id(address="VCU0000001") == 4711
        assert restored.get_interface(address="VCU0000001") == Interface.HMIP_RF
        assert restored.get_channel_rooms(channel_address="VCU0000001:1") == {"Living Room"}
        assert restored.get_device_rooms(device_address="VCU0000001") == {"Living Room"}
        assert restored.get_function_text(address="VCU0000001:1") == "Light"


class TestCentralDataCacheLoad:
    """Test per-interface freshness handling in CentralDataCache.load()."""

    @staticmethod
    def _make_data_cache(*, clients: tuple[MagicMock, ...]) -> CentralDataCache:
        """Create a CentralDataCache with the given mock clients."""
        client_provider = MagicMock()
        client_provider.clients = clients
        central_info = MagicMock()
        central_info.name = "test_central"
        return CentralDataCache(
            device_provider=MagicMock(),
            client_provider=client_provider,
            data_point_provider=MagicMock(),
            central_info=central_info,
        )

    @pytest.mark.asyncio
    async def test_load_fetches_stale_interfaces_even_if_first_is_fresh(self) -> None:
        """A fresh first interface must not prevent fetching the remaining interfaces."""
        client_hmip = MagicMock()
        client_hmip.interface = Interface.HMIP_RF
        client_hmip.fetch_all_device_data = AsyncMock()
        client_cuxd = MagicMock()
        client_cuxd.interface = Interface.CUXD
        client_cuxd.fetch_all_device_data = AsyncMock()

        cache = self._make_data_cache(clients=(client_hmip, client_cuxd))
        # Mark the first client's interface as freshly loaded.
        cache.add_data(interface=Interface.HMIP_RF, all_device_data={"key": "value"})

        await cache.load()

        client_hmip.fetch_all_device_data.assert_not_awaited()
        client_cuxd.fetch_all_device_data.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_load_with_interface_filter_only_fetches_matching_client(self) -> None:
        """An interface filter must restrict the fetch to the matching client."""
        client_hmip = MagicMock()
        client_hmip.interface = Interface.HMIP_RF
        client_hmip.fetch_all_device_data = AsyncMock()
        client_cuxd = MagicMock()
        client_cuxd.interface = Interface.CUXD
        client_cuxd.fetch_all_device_data = AsyncMock()

        cache = self._make_data_cache(clients=(client_hmip, client_cuxd))

        await cache.load(interface=Interface.CUXD)

        client_hmip.fetch_all_device_data.assert_not_awaited()
        client_cuxd.fetch_all_device_data.assert_awaited_once()
