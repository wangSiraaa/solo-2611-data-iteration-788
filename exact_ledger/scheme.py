"""Immutable measurement schemes for joint readings.

A measurement scheme describes the *channels* an instrument can report.  Every
channel is an exact rational linear reading over the scheme's named columns,
with an optional constant:

    channel := constant + sum(coefficient_i * column_i)

For example a bench with columns ``a`` and ``b`` may expose::

    M = a + b
    D = a - b + 1

Schemes are published independently of plans and facts, and -- like plans --
they are immutable.  Re-publishing the same ``scheme_id`` with the same
semantic content returns the stored object; publishing different content under
an existing id is rejected, so a version's channels can never be rewritten in
place.

This module knows nothing about readings, replay or recovery: it only validates
and publishes the channel definitions.  :mod:`exact_ledger.joint` consumes the
published products.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .rational import as_fraction, to_str

__all__ = [
    "MeasurementSchemeError",
    "EmptyIdentifierError",
    "DuplicateChannelError",
    "UnknownColumnError",
    "UnknownChannelError",
    "InvalidSchemeError",
    "Channel",
    "MeasurementScheme",
    "MeasurementSchemeService",
    "MAX_COLUMNS",
    "MAX_CHANNELS",
]

# Documented scale limits for one scheme / one local recovery problem.
MAX_COLUMNS = 8
MAX_CHANNELS = 32


class MeasurementSchemeError(ValueError):
    """Base class for all rejected measurement schemes."""


class EmptyIdentifierError(MeasurementSchemeError):
    pass


class DuplicateChannelError(MeasurementSchemeError):
    pass


class UnknownColumnError(MeasurementSchemeError):
    pass


class UnknownChannelError(MeasurementSchemeError):
    pass


class InvalidSchemeError(MeasurementSchemeError):
    pass


@dataclass(frozen=True)
class Channel:
    """One named linear reading of a scheme.

    ``terms`` preserves the raw declarations (for auditing), while
    ``coefficients`` is the reduced map with zero coefficients removed.  All
    arithmetic is exact.
    """

    channel_id: str
    constant: Fraction
    terms: Tuple[Tuple[str, Fraction], ...]
    coefficients: Mapping[str, Fraction]

    def coefficient_of(self, column: str) -> Fraction:
        return self.coefficients.get(column, Fraction(0))

    def as_dict(self) -> Dict[str, object]:
        return {
            "channel_id": self.channel_id,
            "constant": to_str(self.constant),
            "terms": [
                {"ref": column, "coefficient": to_str(coefficient)}
                for column, coefficient in self.terms
            ],
        }

    def semantic_dict(self) -> Dict[str, object]:
        """Canonical, order-insensitive description used for the digest."""
        return {
            "channel_id": self.channel_id,
            "constant": to_str(self.constant),
            "coefficients": {
                column: to_str(coefficient)
                for column, coefficient in sorted(self.coefficients.items())
            },
        }


@dataclass(frozen=True)
class MeasurementScheme:
    """An immutable, published measurement scheme."""

    scheme_id: str
    columns: Tuple[str, ...]
    channels: Tuple[Channel, ...]
    channel_map: Mapping[str, Channel]
    digest: str

    def channel(self, channel_id: str) -> Channel:
        try:
            return self.channel_map[channel_id]
        except KeyError as exc:
            raise UnknownChannelError(
                f"scheme {self.scheme_id!r} has no channel {channel_id!r}"
            ) from exc

    def has_channel(self, channel_id: str) -> bool:
        return channel_id in self.channel_map

    def column_set(self) -> frozenset:
        return frozenset(self.columns)

    def as_dict(self) -> Dict[str, object]:
        """The publishable definition (no derived digest)."""
        return {
            "scheme_id": self.scheme_id,
            "columns": list(self.columns),
            "channels": [channel.as_dict() for channel in self.channels],
        }

    def as_explain_dict(self) -> Dict[str, object]:
        return {
            "scheme_id": self.scheme_id,
            "digest": self.digest,
            "columns": list(self.columns),
            "channels": {
                channel.channel_id: {
                    "constant": to_str(channel.constant),
                    "coefficients": {
                        column: to_str(coefficient)
                        for column, coefficient in channel.coefficients.items()
                    },
                    "terms": channel.as_dict()["terms"],
                }
                for channel in self.channels
            },
        }


def _require_identifier(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EmptyIdentifierError(
            f"{what} must be a non-empty string, got {value!r}"
        )
    return value


def _check_coefficient(value: object, what: str) -> Fraction:
    try:
        return as_fraction(value)  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - re-raised as a scheme error
        raise InvalidSchemeError(f"{what}: {exc}") from exc


def _parse_terms(raw_terms: object, columns: set, what: str) -> Tuple[Tuple[str, Fraction], ...]:
    if raw_terms is None:
        return ()
    if not isinstance(raw_terms, (list, tuple)):
        raise InvalidSchemeError(f"{what}: 'terms' must be a list")
    terms: List[Tuple[str, Fraction]] = []
    for index, raw in enumerate(raw_terms):
        if not isinstance(raw, Mapping):
            raise InvalidSchemeError(f"{what}: term {index} must be an object")
        ref = raw.get("ref", raw.get("column"))
        if ref is None:
            raise InvalidSchemeError(f"{what}: term {index} is missing 'ref'")
        column = _require_identifier(ref, f"{what} term {index} ref")
        if column not in columns:
            raise UnknownColumnError(
                f"{what} term {index} references unknown column {column!r}; "
                f"declared columns: {sorted(columns)}"
            )
        coefficient = _check_coefficient(
            raw.get("coefficient", raw.get("coef", "1")),
            f"{what} term {index} coefficient",
        )
        terms.append((column, coefficient))
    return tuple(terms)


def _parse_channel(
    raw: object, columns: set, what: str, fallback_id: Optional[str] = None
) -> Channel:
    if not isinstance(raw, Mapping):
        raise InvalidSchemeError(f"{what} must be an object")
    channel_id = raw.get("channel_id", raw.get("name", raw.get("id", fallback_id)))
    channel_id = _require_identifier(channel_id, f"{what} channel_id")
    terms = _parse_terms(raw.get("terms", raw.get("refs", [])), columns, what)
    constant = _check_coefficient(raw.get("constant", "0"), f"{what} constant")

    reduced: Dict[str, Fraction] = {}
    for column, coefficient in terms:
        reduced[column] = reduced.get(column, Fraction(0)) + coefficient
    reduced = {
        column: coefficient
        for column, coefficient in reduced.items()
        if coefficient != 0
    }
    return Channel(
        channel_id=channel_id,
        constant=constant,
        terms=terms,
        coefficients=MappingProxyType(reduced),
    )


class MeasurementSchemeService:
    """Publishes immutable measurement schemes.

    The service only owns scheme definitions; readings, plans and recovery live
    elsewhere.
    """

    def __init__(self) -> None:
        self._schemes: Dict[str, MeasurementScheme] = {}

    # ------------------------------------------------------------------ API
    def publish(self, definition: Mapping[str, object]) -> MeasurementScheme:
        if not isinstance(definition, Mapping):
            raise InvalidSchemeError("measurement scheme must be an object")
        scheme_id = _require_identifier(definition.get("scheme_id"), "scheme_id")

        raw_columns = definition.get("columns", [])
        if not isinstance(raw_columns, (list, tuple)):
            raise InvalidSchemeError("scheme columns must be a list")
        columns: List[str] = []
        seen_columns = set()
        for raw in raw_columns:
            column = _require_identifier(raw, "column name")
            if column in seen_columns:
                raise DuplicateChannelError(f"duplicate column {column!r}")
            seen_columns.add(column)
            columns.append(column)
        if len(columns) > MAX_COLUMNS:
            raise InvalidSchemeError(
                f"scheme has {len(columns)} columns, limit is {MAX_COLUMNS}"
            )
        column_set = set(columns)

        raw_channels = definition.get("channels", [])
        if isinstance(raw_channels, Mapping):
            items: List[Tuple[str, object]] = list(raw_channels.items())
        elif isinstance(raw_channels, (list, tuple)):
            items = [(None, raw) for raw in raw_channels]  # type: ignore[list-item]
        else:
            raise InvalidSchemeError("scheme channels must be a list or object")

        channels: List[Channel] = []
        seen_channels: Dict[str, Channel] = {}
        for index, (fallback_id, raw) in enumerate(items):
            channel = _parse_channel(
                raw, column_set, f"channel {index}", fallback_id=fallback_id
            )
            if channel.channel_id in seen_channels:
                raise DuplicateChannelError(
                    f"duplicate channel id {channel.channel_id!r}"
                )
            seen_channels[channel.channel_id] = channel
            channels.append(channel)
        if len(channels) > MAX_CHANNELS:
            raise InvalidSchemeError(
                f"scheme has {len(channels)} channels, limit is {MAX_CHANNELS}"
            )

        digest = self._digest(scheme_id, columns, channels)
        existing = self._schemes.get(scheme_id)
        if existing is not None:
            if existing.digest != digest:
                raise InvalidSchemeError(
                    f"scheme_id {scheme_id!r} is already published with "
                    "different content; published schemes are immutable "
                    "(publish a new scheme version instead of rewriting a "
                    "channel in place)"
                )
            return existing

        scheme = MeasurementScheme(
            scheme_id=scheme_id,
            columns=tuple(columns),
            channels=tuple(channels),
            channel_map=MappingProxyType(dict(seen_channels)),
            digest=digest,
        )
        self._schemes[scheme_id] = scheme
        return scheme

    def get(self, scheme_id: str) -> MeasurementScheme:
        try:
            return self._schemes[scheme_id]
        except KeyError as exc:
            raise UnknownChannelError(
                f"unknown measurement scheme {scheme_id!r}"
            ) from exc

    def has(self, scheme_id: str) -> bool:
        return scheme_id in self._schemes

    def all_schemes(self) -> Tuple[MeasurementScheme, ...]:
        return tuple(self._schemes.values())

    def explain(
        self, scheme_id: str, channel_id: Optional[str] = None
    ) -> Dict[str, object]:
        scheme = self.get(scheme_id)
        if channel_id is None:
            return scheme.as_explain_dict()
        channel = scheme.channel(channel_id)
        return {
            "scheme_id": scheme.scheme_id,
            "digest": scheme.digest,
            "channel_id": channel.channel_id,
            "constant": to_str(channel.constant),
            "coefficients": {
                column: to_str(coefficient)
                for column, coefficient in channel.coefficients.items()
            },
            "terms": channel.as_dict()["terms"],
        }

    # ------------------------------------------------------------- internals
    @staticmethod
    def _digest(
        scheme_id: str, columns: Sequence[str], channels: Sequence[Channel]
    ) -> str:
        payload = {
            "scheme_id": scheme_id,
            "columns": sorted(columns),
            "channels": [
                channel.semantic_dict()
                for channel in sorted(channels, key=lambda item: item.channel_id)
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
