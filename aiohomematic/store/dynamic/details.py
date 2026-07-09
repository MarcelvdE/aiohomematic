# SPDX-License-Identifier: MIT
# Copyright (c) 2021-2026
"""
Device details cache for runtime device metadata.

This module provides DeviceDetailsCache which enriches devices with human-readable
names, interface mapping, rooms, functions, and address IDs fetched via the backend.

The fetch behind ``refresh()`` is expensive for the CCU: ``Device.listAllDetail``,
``Room.getAll`` and ``Subsection.getAll`` are CCU-wide JSON-RPC calls without any
filter parameter. The cache is therefore persisted to disk (warm restarts reuse the
last known metadata and refresh in the background) and guarded by a long freshness
TTL (``DEVICE_DETAILS_MAX_CACHE_AGE``) because names/rooms/functions rarely change.
"""

import asyncio
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime
import logging
from typing import Any, Final, cast

from aiohomematic.const import DEVICE_DETAILS_MAX_CACHE_AGE, INIT_DATETIME, DataOperationResult, Interface
from aiohomematic.interfaces import (
    CentralInfoProtocol,
    ConfigProviderProtocol,
    DeviceDetailsProviderProtocol,
    DeviceDetailsWriterProtocol,
    PrimaryClientProviderProtocol,
)
from aiohomematic.interfaces.model import DeviceRemovalInfoProtocol
from aiohomematic.property_decorators import DelegatedProperty
from aiohomematic.store.persistent.base import BasePersistentCache
from aiohomematic.store.storage import StorageProtocol
from aiohomematic.support import changed_within_seconds
from aiohomematic.support.address import get_device_address

_LOGGER: Final = logging.getLogger(__name__)

_CHANNEL_ROOMS: Final = "channel_rooms"
_FUNCTIONS: Final = "functions"
_INTERFACES: Final = "interfaces"
_ISE_IDS: Final = "ise_ids"
_NAMES: Final = "names"


class DeviceDetailsCache(BasePersistentCache, DeviceDetailsProviderProtocol, DeviceDetailsWriterProtocol):
    """
    Cache for device/channel details.

    Persistence
    -----------
    The cache content (names, ReGa ids, interfaces, rooms, functions) is persisted
    to disk via the storage abstraction. On a warm restart the persisted content is
    loaded via ``load()`` so device creation does not have to wait for the CCU-wide
    ``Device.listAllDetail``/``Room.getAll``/``Subsection.getAll`` fetch; a
    background ``refresh()`` afterwards brings the cache up to date.

    Concurrency
    -----------
    This class assumes single asyncio event-loop execution. All dictionary
    operations are atomic under cooperative multitasking (no preemption
    between synchronous instructions). Composite operations (check-then-update)
    are safe because no ``await`` occurs between the check and the mutation.

    If Python free-threading (PEP 703) is adopted, these operations will
    need ``asyncio.Lock`` protection for composite sequences.
    """

    __slots__ = (
        "_central_info",
        "_channel_rooms",
        "_device_channel_ise_ids",
        "_device_rooms",
        "_functions",
        "_interface_cache",
        "_names_cache",
        "_primary_client_provider",
        "_refresh_lock",
        "_refreshed_at",
    )

    def __init__(
        self,
        *,
        central_info: CentralInfoProtocol,
        config_provider: ConfigProviderProtocol,
        primary_client_provider: PrimaryClientProviderProtocol,
        storage: StorageProtocol,
    ) -> None:
        """Initialize the device details cache."""
        super().__init__(storage=storage, config_provider=config_provider)
        self._central_info: Final = central_info
        self._primary_client_provider: Final = primary_client_provider
        self._channel_rooms: Final[dict[str, set[str]]] = defaultdict(set)
        self._device_channel_ise_ids: Final[dict[str, int]] = {}
        self._device_rooms: Final[dict[str, set[str]]] = defaultdict(set)
        self._functions: Final[dict[str, set[str]]] = {}
        self._interface_cache: Final[dict[str, Interface]] = {}
        self._refresh_lock: Final = asyncio.Lock()
        self._names_cache: Final[dict[str, str]] = {}
        self._refreshed_at = INIT_DATETIME

    device_channel_ise_ids: Final = DelegatedProperty[Mapping[str, int]](path="_device_channel_ise_ids")

    def add_address_ise_id(self, *, address: str, ise_id: int) -> None:
        """Add channel id for a channel."""
        self._device_channel_ise_ids[address] = ise_id

    def add_interface(self, *, address: str, interface: Interface) -> None:
        """Add interface to cache."""
        self._interface_cache[address] = interface

    def add_name(self, *, address: str, name: str) -> None:
        """Add name to cache."""
        self._names_cache[address] = name

    async def clear(self) -> None:
        """Remove persisted content and clear the in-memory cache."""
        await super().clear()
        self.clear_in_memory()

    def clear_in_memory(self) -> None:
        """Clear the in-memory cache content."""
        self._names_cache.clear()
        self._channel_rooms.clear()
        self._device_rooms.clear()
        self._functions.clear()
        self._refreshed_at = INIT_DATETIME

    def get_address_id(self, *, address: str) -> int:
        """Get id for address."""
        return self._device_channel_ise_ids.get(address) or 0

    def get_channel_rooms(self, *, channel_address: str) -> set[str]:
        """Return rooms by channel_address."""
        return self._channel_rooms[channel_address]

    def get_device_rooms(self, *, device_address: str) -> set[str]:
        """Return all rooms by device_address."""
        return set(self._device_rooms.get(device_address, ()))

    def get_function_text(self, *, address: str) -> str | None:
        """Return function by address."""
        if functions := self._functions.get(address):
            return ",".join(functions)
        return None

    def get_interface(self, *, address: str) -> Interface:
        """Get interface from cache."""
        return self._interface_cache.get(address) or Interface.BIDCOS_RF

    def get_name(self, *, address: str) -> str | None:
        """Get name from cache."""
        return self._names_cache.get(address)

    async def refresh(self, *, direct_call: bool = False) -> None:
        """
        Fetch names, rooms and functions from the backend.

        No-ops while the cache is fresh (``DEVICE_DETAILS_MAX_CACHE_AGE``) unless
        ``direct_call`` is True. The fetch is CCU-wide (``Device.listAllDetail``,
        ``Room.getAll``, ``Subsection.getAll`` have no filter parameter), so callers
        must not force it more often than necessary.

        Serialized via ``_refresh_lock``: multiple callers (e.g. the per-interface
        scheduled refresh, which fans out over all clients concurrently via
        asyncio.gather) can race the check-then-refetch sequence below, since it
        awaits between the staleness check and repopulating the cache.

        The fetched data is swapped into the cache without an intermediate cleared
        state: names are overwritten in place, rooms/functions are collected first
        and replaced synchronously. Devices created concurrently therefore never
        observe a half-empty cache.
        """
        async with self._refresh_lock:
            if direct_call is False and changed_within_seconds(
                last_change=self._refreshed_at, max_age=DEVICE_DETAILS_MAX_CACHE_AGE
            ):
                return
            if (client := self._primary_client_provider.primary_client) is None:
                # Keep whatever is cached; a later call retries once a client exists.
                return
            _LOGGER.debug("REFRESH: Loading names for %s", self._central_info.name)
            await client.fetch_device_details()
            _LOGGER.debug("REFRESH: Loading rooms for %s", self._central_info.name)
            channel_rooms = await self._get_all_rooms()
            _LOGGER.debug("REFRESH: Loading functions for %s", self._central_info.name)
            functions = await self._get_all_functions()
            # No await between here and the end of the block: replace atomically.
            self._channel_rooms.clear()
            self._channel_rooms.update(channel_rooms)
            self._device_rooms.clear()
            self._device_rooms.update(self._prepare_device_rooms())
            self._functions.clear()
            self._functions.update(functions)
            self._refreshed_at = datetime.now()
        await self.save()

    def remove_device(self, *, device: DeviceRemovalInfoProtocol) -> None:
        """Remove device data from all caches."""
        # Clean device-level entries
        self._names_cache.pop(device.address, None)
        self._interface_cache.pop(device.address, None)
        self._device_channel_ise_ids.pop(device.address, None)
        self._device_rooms.pop(device.address, None)
        self._functions.pop(device.address, None)

        # Clean channel-level entries
        for channel_address in device.channels:
            self._names_cache.pop(channel_address, None)
            self._interface_cache.pop(channel_address, None)
            self._device_channel_ise_ids.pop(channel_address, None)
            self._channel_rooms.pop(channel_address, None)
            self._functions.pop(channel_address, None)

    async def save(self) -> DataOperationResult:
        """Persist the device details to storage."""
        self._content.clear()
        self._content.update(
            {
                _NAMES: dict(self._names_cache),
                _ISE_IDS: dict(self._device_channel_ise_ids),
                _INTERFACES: {address: interface.value for address, interface in self._interface_cache.items()},
                _CHANNEL_ROOMS: {address: sorted(rooms) for address, rooms in self._channel_rooms.items() if rooms},
                _FUNCTIONS: {address: sorted(functions) for address, functions in self._functions.items()},
            }
        )
        return await super().save()

    def _create_empty_content(self) -> dict[str, Any]:
        """Create empty content structure."""
        return {
            _NAMES: {},
            _ISE_IDS: {},
            _INTERFACES: {},
            _CHANNEL_ROOMS: {},
            _FUNCTIONS: {},
        }

    async def _get_all_functions(self) -> Mapping[str, set[str]]:
        """Get all functions, if available."""
        if client := self._primary_client_provider.primary_client:
            return cast(
                Mapping[str, set[str]],
                await client.get_all_functions(),
            )
        return {}

    async def _get_all_rooms(self) -> Mapping[str, set[str]]:
        """Get all rooms, if available."""
        if client := self._primary_client_provider.primary_client:
            return cast(
                Mapping[str, set[str]],
                await client.get_all_rooms(),
            )
        return {}

    def _prepare_device_rooms(self) -> dict[str, set[str]]:
        """
        Return rooms by device_address.

        Aggregation algorithm:
            The CCU stores room assignments at the channel level (e.g., "ABC123:1" is in "Living Room").
            Devices themselves don't have direct room assignments - they inherit from their channels.
            This method aggregates channel rooms to the device level by:
            1. Iterating all channel_address -> rooms mappings
            2. Extracting the device_address from each channel_address
            3. Merging all channel rooms into a set per device

        Result: A device is considered "in" all rooms that any of its channels are in.
        """
        _device_rooms: Final[dict[str, set[str]]] = defaultdict(set)
        for channel_address, rooms in self._channel_rooms.items():
            if rooms:
                # Extract device address (e.g., "ABC123:1" -> "ABC123")
                # and merge this channel's rooms into the device's room set
                _device_rooms[get_device_address(address=channel_address)].update(rooms)
        return _device_rooms

    def _process_loaded_content(self, *, data: dict[str, Any]) -> None:
        """Rebuild the in-memory caches from persisted content."""
        self._names_cache.clear()
        self._names_cache.update(data.get(_NAMES, {}))
        self._device_channel_ise_ids.clear()
        self._device_channel_ise_ids.update(data.get(_ISE_IDS, {}))
        self._interface_cache.clear()
        self._interface_cache.update(
            {
                address: Interface(interface)
                for address, interface in data.get(_INTERFACES, {}).items()
                if interface in Interface
            }
        )
        self._channel_rooms.clear()
        for address, rooms in data.get(_CHANNEL_ROOMS, {}).items():
            self._channel_rooms[address] = set(rooms)
        self._device_rooms.clear()
        self._device_rooms.update(self._prepare_device_rooms())
        self._functions.clear()
        self._functions.update({address: set(functions) for address, functions in data.get(_FUNCTIONS, {}).items()})
        # Persisted content is a warm-start aid, not fresh data: leave _refreshed_at
        # at INIT_DATETIME so the next refresh() actually fetches from the backend.
