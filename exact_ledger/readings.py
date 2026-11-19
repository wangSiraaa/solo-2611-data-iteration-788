"""Versioned joint-reading storage (SQLite).

A reading is identified by ``(record_id, scheme_id, channel_id, revision)``.
The *value* is the scalar the instrument reported for that channel; the channel
definition (an exact rational linear form) lives in
:mod:`exact_ledger.scheme`.  Keeping the identity a revision is what makes the
little import rules meaningful:

* the highest revision of an identity is the effective one;
* a lower revision that arrives **late** is stored but ignored by the effective
  view -- it never overwrites newer data and it is never discarded;
* re-importing the same identity with the *same* canonical value is an
  idempotent replay;
* importing a different value for an already-stored identity rejects the whole
  batch and writes nothing (including the other rows of that batch);
* a channel may be **retracted** in a higher revision (a tombstone); a yet
  higher revision with a value restores it.

Different channels may genuinely disagree -- that is a *measurement* conflict,
not an import error.  The store keeps every channel's original reading so the
conflict can be traced to its sources; it never silently keeps "the last one".
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import sqlite3
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .rational import as_fraction, to_str

__all__ = [
    "ReadingError",
    "InvalidReadingError",
    "ReadingConflictError",
    "UnknownReadingError",
    "Reading",
    "ReadingStore",
    "MAX_READINGS_PER_PROBLEM",
]

# Documented bound: at most 32 channels (hence readings) per local problem.
MAX_READINGS_PER_PROBLEM = 32


class ReadingError(ValueError):
    """Base class for rejected reading imports."""


class InvalidReadingError(ReadingError):
    pass


class ReadingConflictError(ReadingError):
    """Same identity with different canonical content (or conflicting batch)."""


class UnknownReadingError(ReadingError):
    pass


def _require_name(value: object, what: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InvalidReadingError(
            f"{what} must be a non-empty string, got {value!r}"
        )
    return value


def _require_revision(value: object) -> int:
    if isinstance(value, bool):
        raise InvalidReadingError("revision must be a positive integer, not a boolean")
    if isinstance(value, str):
        if not value.strip().isdigit():
            raise InvalidReadingError(
                f"revision must be a positive integer, got {value!r}"
            )
        value = int(value.strip())
    if not isinstance(value, int) or value < 1:
        raise InvalidReadingError(f"revision must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class Reading:
    record_id: str
    scheme_id: str
    channel_id: str
    revision: int
    value: Optional[Fraction]  # None means retracted (tombstone)
    imported_seq: int = 0

    @property
    def retracted(self) -> bool:
        return self.value is None

    def canonical_value(self) -> Optional[str]:
        return None if self.value is None else to_str(self.value)

    def as_dict(self) -> Dict[str, object]:
        return {
            "record_id": self.record_id,
            "scheme_id": self.scheme_id,
            "channel_id": self.channel_id,
            "revision": self.revision,
            "value": self.canonical_value(),
            "retracted": self.retracted,
        }


class ReadingStore:
    """Append-only, SQLite-backed versioned reading store."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS joint_readings (
                record_id TEXT NOT NULL,
                scheme_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                is_retracted INTEGER NOT NULL,
                value_num TEXT,
                value_den TEXT,
                imported_seq INTEGER NOT NULL,
                PRIMARY KEY (record_id, scheme_id, channel_id, revision)
            );
            CREATE INDEX IF NOT EXISTS joint_readings_lookup
                ON joint_readings (record_id, scheme_id, channel_id, revision);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------- reading
    def _load(self, row: sqlite3.Row) -> Reading:
        value = None
        if not row["is_retracted"]:
            value = Fraction(int(row["value_num"]), int(row["value_den"]))
        return Reading(
            record_id=row["record_id"],
            scheme_id=row["scheme_id"],
            channel_id=row["channel_id"],
            revision=row["revision"],
            value=value,
            imported_seq=row["imported_seq"],
        )

    def all_revisions(
        self, record_id: str, scheme_id: str, channel_id: str
    ) -> List[Reading]:
        rows = self._conn.execute(
            "SELECT * FROM joint_readings WHERE record_id = ? AND scheme_id = ? "
            "AND channel_id = ? ORDER BY revision",
            (record_id, scheme_id, channel_id),
        ).fetchall()
        return [self._load(row) for row in rows]

    def get_revision(
        self, record_id: str, scheme_id: str, channel_id: str, revision: int
    ) -> Optional[Reading]:
        row = self._conn.execute(
            "SELECT * FROM joint_readings WHERE record_id = ? AND scheme_id = ? "
            "AND channel_id = ? AND revision = ?",
            (record_id, scheme_id, channel_id, revision),
        ).fetchone()
        return self._load(row) if row else None

    def _effective_rows(
        self, record_id: Optional[str] = None, scheme_id: Optional[str] = None
    ) -> List[sqlite3.Row]:
        clauses = []
        params: List[object] = []
        if record_id is not None:
            clauses.append("record_id = ?")
            params.append(record_id)
        if scheme_id is not None:
            clauses.append("scheme_id = ?")
            params.append(scheme_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        return self._conn.execute(
            f"""
            SELECT r.* FROM joint_readings r
            JOIN (
                SELECT record_id, scheme_id, channel_id, MAX(revision) AS revision
                FROM joint_readings {where}
                GROUP BY record_id, scheme_id, channel_id
            ) latest
              ON r.record_id = latest.record_id
             AND r.scheme_id = latest.scheme_id
             AND r.channel_id = latest.channel_id
             AND r.revision = latest.revision
            ORDER BY r.record_id, r.scheme_id, r.channel_id
            """,
            params,
        ).fetchall()

    def effective(
        self, record_id: Optional[str] = None, scheme_id: Optional[str] = None
    ) -> Dict[Tuple[str, str, str], Reading]:
        """Effective reading per identity; retracted channels are excluded."""
        result: Dict[Tuple[str, str, str], Reading] = {}
        for row in self._effective_rows(record_id, scheme_id):
            reading = self._load(row)
            if reading.retracted:
                continue
            result[(reading.record_id, reading.scheme_id, reading.channel_id)] = reading
        return result

    def effective_for(self, record_id: str, scheme_id: str) -> Tuple[Reading, ...]:
        rows = [
            reading
            for key, reading in self.effective(record_id, scheme_id).items()
        ]
        rows.sort(key=lambda item: item.channel_id)
        return tuple(rows)

    def record_ids(self) -> List[str]:
        return [
            row["record_id"]
            for row in self._conn.execute(
                "SELECT DISTINCT record_id FROM joint_readings ORDER BY record_id"
            )
        ]

    def version_signature(self, record_id: str, scheme_id: str) -> Tuple:
        """Compact signature of the effective version (for solver caches)."""
        return tuple(
            (reading.channel_id, reading.revision, reading.canonical_value())
            for reading in self.effective_for(record_id, scheme_id)
        )

    def snapshot(self, record_id: str, scheme_id: str) -> List[Dict[str, object]]:
        return [
            reading.as_dict()
            for reading in self.effective_for(record_id, scheme_id)
        ]

    # ------------------------------------------------------------- writing
    def prepare(self, raw_readings: Sequence[object]) -> List[Reading]:
        """Validate a whole batch without writing anything."""
        prepared: List[Reading] = []
        batch_keys: Dict[Tuple[str, str, str, int], Reading] = {}
        for index, raw in enumerate(raw_readings):
            reading = self._parse(raw, index)
            key = (
                reading.record_id,
                reading.scheme_id,
                reading.channel_id,
                reading.revision,
            )
            previous = batch_keys.get(key)
            if previous is not None:
                if previous.canonical_value() != reading.canonical_value():
                    raise ReadingConflictError(
                        f"batch reports identity {key} with two different "
                        f"readings ({previous.canonical_value()!r} vs "
                        f"{reading.canonical_value()!r}); the entire batch is "
                        "rejected"
                    )
                continue
            batch_keys[key] = reading
            prepared.append(reading)

        for reading in prepared:
            existing = self.get_revision(
                reading.record_id,
                reading.scheme_id,
                reading.channel_id,
                reading.revision,
            )
            if existing is not None and (
                existing.canonical_value() != reading.canonical_value()
            ):
                raise ReadingConflictError(
                    f"identity ({reading.record_id!r}, {reading.scheme_id!r}, "
                    f"{reading.channel_id!r}, revision {reading.revision}) "
                    f"already exists with reading {existing.canonical_value()!r}; "
                    "the entire batch is rejected"
                )
        return prepared

    def _parse(self, raw: object, index: int) -> Reading:
        if not isinstance(raw, Mapping):
            raise InvalidReadingError(f"reading {index} must be an object")
        record_id = _require_name(raw.get("record_id"), f"reading {index} record_id")
        scheme_id = _require_name(raw.get("scheme_id"), f"reading {index} scheme_id")
        channel_id = _require_name(
            raw.get("channel_id"), f"reading {index} channel_id"
        )
        revision = _require_revision(raw.get("revision"))
        retract = raw.get("retract", False)
        if not isinstance(retract, bool):
            raise InvalidReadingError(
                f"reading {index} 'retract' must be a boolean"
            )
        if retract:
            if raw.get("value") is not None:
                raise InvalidReadingError(
                    f"reading {index} retracts a channel but also supplies a value"
                )
            value: Optional[Fraction] = None
        else:
            if "value" not in raw or raw["value"] is None:
                raise InvalidReadingError(
                    f"reading {index} is missing a value (omit only when retracting)"
                )
            try:
                value = as_fraction(raw["value"])
            except Exception as exc:  # noqa: BLE001
                raise InvalidReadingError(
                    f"reading {index} value: {exc}"
                ) from exc
        return Reading(
            record_id=record_id,
            scheme_id=scheme_id,
            channel_id=channel_id,
            revision=revision,
            value=value,
        )

    def write(self, readings: Sequence[Reading]) -> int:
        added = 0
        with self._conn:
            seq = self._conn.execute(
                "SELECT COALESCE(MAX(imported_seq), 0) FROM joint_readings"
            ).fetchone()[0]
            for reading in readings:
                if self.get_revision(
                    reading.record_id,
                    reading.scheme_id,
                    reading.channel_id,
                    reading.revision,
                ) is not None:
                    continue
                seq += 1
                value = reading.value
                self._conn.execute(
                    "INSERT INTO joint_readings (record_id, scheme_id, "
                    "channel_id, revision, is_retracted, value_num, value_den, "
                    "imported_seq) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        reading.record_id,
                        reading.scheme_id,
                        reading.channel_id,
                        reading.revision,
                        int(reading.retracted),
                        None if value is None else str(value.numerator),
                        None if value is None else str(value.denominator),
                        seq,
                    ),
                )
                added += 1
        return added

    def import_readings(self, raw_readings: Sequence[object]) -> int:
        prepared = self.prepare(raw_readings)
        return self.write(prepared)
