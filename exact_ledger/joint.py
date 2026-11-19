"""Joint-reading recovery: per-record identifiability proofs.

Readings from a :mod:`exact_ledger.scheme` scheme are *linear observations*::

    channel := constant + sum(coefficient * column)  =  value

A record can be under-determined (``a + b = 7`` says nothing about ``a``),
redundant (a second reading consistent with the first), or contradictory
(different channels really disagree).  This module consumes the current
effective readings and the *published* plan contracts and decides, per record
and per requested output:

``decided``    the output has one exact rational value under every solution of
               the readings; a witness expresses the value as an exact
               combination of the reading equations that can be substituted
               back independently.
``nonunique``  two concrete solutions of *all* readings are returned whose
               output values differ.
``conflict``   the readings are inconsistent; a set of reading equations is
               returned whose exact combination is ``0 = nonzero``.

Targets are decided independently: one undetermined column never poisons a
different output that is still identifiable, and a missing column is never
guessed from a default.  A judgement is persisted as an immutable proof bound
to the reading version, the scheme digest and the target plan digest, so it can
be reopened (and re-verified) after the database is closed and reopened.

Nothing here writes or rewrites a plan or a scheme; those remain the sources of
truth for the channel definitions and the output contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import sqlite3
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .plan import MAX_OUTPUTS, Plan, PlanService
from .rational import as_fraction, to_str
from .readings import Reading, ReadingError, ReadingStore
from .scheme import (
    MAX_COLUMNS,
    Channel,
    MeasurementScheme,
    MeasurementSchemeService,
    UnknownChannelError,
)

__all__ = [
    "JointError",
    "UnknownRecordError",
    "UnknownSchemeError",
    "UnknownProofError",
    "DuplicateProofError",
    "InvalidTargetError",
    "Proof",
    "JointService",
    "open_joint",
]

CONSTANT_INPUT = ""


class JointError(ValueError):
    pass


class UnknownRecordError(JointError):
    pass


class UnknownSchemeError(JointError):
    pass


class UnknownProofError(JointError):
    pass


class DuplicateProofError(JointError):
    pass


class InvalidTargetError(JointError):
    pass


# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Proof:
    """An immutable, reopened joint-recovery proof."""

    proof_id: str
    record_id: str
    scheme_id: str
    scheme_digest: str
    target_plan_id: str
    target_plan_digest: str
    requested_outputs: Tuple[str, ...]
    status: str
    evaluation: Mapping[str, object]
    readings_snapshot: Tuple[Mapping[str, object], ...]
    verification: Mapping[str, object]
    created_seq: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "proof_id": self.proof_id,
            "record_id": self.record_id,
            "scheme_id": self.scheme_id,
            "scheme_digest": self.scheme_digest,
            "target_plan_id": self.target_plan_id,
            "target_plan_digest": self.target_plan_digest,
            "requested_outputs": list(self.requested_outputs),
            "status": self.status,
            "evaluation": dict(self.evaluation),
            "readings_snapshot": [dict(item) for item in self.readings_snapshot],
            "verification": dict(self.verification),
            "created_seq": self.created_seq,
        }

    def target(self, output: str) -> Mapping[str, object]:
        for entry in self.evaluation.get("targets", []):  # type: ignore[union-attr]
            if entry["output"] == output:
                return entry
        raise InvalidTargetError(
            f"proof {self.proof_id!r} has no target {output!r}"
        )


# --------------------------------------------------------------------------
def _f(value: Fraction) -> str:
    return to_str(value)


def _nonzero(mapping: Mapping[str, Fraction]) -> Dict[str, Fraction]:
    return {key: value for key, value in mapping.items() if value != 0}


def _reading_ref(
    index: int,
    reading: Reading,
    scheme: MeasurementScheme,
) -> Dict[str, object]:
    channel = scheme.channel(reading.channel_id)
    return {
        "index": index,
        "record_id": reading.record_id,
        "scheme_id": reading.scheme_id,
        "channel_id": reading.channel_id,
        "revision": reading.revision,
        "value": _f(reading.value),  # type: ignore[arg-type]
        "constant": _f(channel.constant),
        "coefficients": {
            column: _f(coefficient)
            for column, coefficient in channel.coefficients.items()
        },
    }


def _equation_dict(reading: Reading, channel: Channel) -> Dict[str, object]:
    direct = None
    if len(channel.coefficients) == 1:
        direct = next(iter(channel.coefficients))
    return {
        "record_id": reading.record_id,
        "scheme_id": reading.scheme_id,
        "channel_id": reading.channel_id,
        "revision": reading.revision,
        "value": _f(reading.value),  # type: ignore[arg-type]
        "constant": _f(channel.constant),
        "coefficients": {
            column: _f(coefficient)
            for column, coefficient in channel.coefficients.items()
        },
        "direct_column": direct,
    }


# --------------------------------------------------------------------------
class _System:
    """Reduced row-echelon form of a set of reading equations.

    Each row is ``[coefficients, rhs, source]`` where ``source`` maps a reading
    index to the exact multiplier with which that reading's equation composes
    the row.  Tracking the source is what lets a contradiction or a value be
    reported as a checkable combination of *original* readings rather than an
    opaque normal-form artefact.
    """

    def __init__(
        self,
        rows: Sequence[Tuple[Mapping[str, Fraction], Fraction]],
        columns: Sequence[str],
    ) -> None:
        self.columns = tuple(columns)
        self.consistent = True
        self.pivot_cols: List[str] = []
        self.reduced: List[Tuple[Dict[str, Fraction], Fraction, Dict[int, Fraction]]] = []
        self.rows: List[Tuple[Dict[str, Fraction], Fraction, Dict[int, Fraction]]] = []
        self.conflict_source: Optional[Dict[int, Fraction]] = None
        self.conflict_residual: Optional[Fraction] = None
        mat = [
            [dict(_nonzero(coeffs)), rhs, {index: Fraction(1)}]
            for index, (coeffs, rhs) in enumerate(rows)
        ]
        pivot = 0
        for column in self.columns:
            found = None
            for index in range(pivot, len(mat)):
                if mat[index][0].get(column):
                    found = index
                    break
            if found is None:
                continue
            mat[pivot], mat[found] = mat[found], mat[pivot]
            coeffs, rhs, source = mat[pivot]
            scale = Fraction(1) / coeffs[column]
            mat[pivot][0] = _nonzero(
                {key: value * scale for key, value in coeffs.items()}
            )
            mat[pivot][1] = rhs * scale
            mat[pivot][2] = _nonzero(
                {key: value * scale for key, value in source.items()}
            )
            coeffs, rhs, source = mat[pivot]
            for index in range(len(mat)):
                if index == pivot:
                    continue
                factor = mat[index][0].get(column)
                if not factor:
                    continue
                other_coeffs, other_rhs, other_source = mat[index]
                new_coeffs = dict(other_coeffs)
                for key, value in coeffs.items():
                    new_coeffs[key] = new_coeffs.get(key, Fraction(0)) - factor * value
                new_source = dict(other_source)
                for key, value in source.items():
                    new_source[key] = new_source.get(key, Fraction(0)) - factor * value
                mat[index] = [
                    _nonzero(new_coeffs),
                    other_rhs - factor * rhs,
                    _nonzero(new_source),
                ]
            self.pivot_cols.append(column)
            pivot += 1
            if pivot == len(mat):
                break
        self.rows = [
            (coeffs, rhs, source) for coeffs, rhs, source in mat
        ]
        for coeffs, rhs, source in mat:
            if not coeffs and rhs != 0:
                self.consistent = False
                self.conflict_source = source
                self.conflict_residual = rhs
                break
        self.reduced = [
            (coeffs, rhs, source) for coeffs, rhs, source in mat[:pivot]
        ]

    def free_columns(self) -> List[str]:
        pivots = set(self.pivot_cols)
        return [column for column in self.columns if column not in pivots]

    def combine(
        self, target_vector: Mapping[str, Fraction]
    ) -> Tuple[Dict[str, Fraction], Fraction, List[Tuple[Fraction, Tuple]]]:
        """Express ``target_vector · x`` via the reduced rows.

        Returns ``(residual, constant, used)``.  ``residual`` collects whatever
        dependence on free columns remains (empty means the target is fixed);
        ``constant`` is the exact target value when it is fixed.
        """
        residual = _nonzero(dict(target_vector))
        constant = Fraction(0)
        used: List[Tuple[Fraction, Tuple]] = []
        for row, pivot in zip(self.reduced, self.pivot_cols):
            coeffs, rhs, source = row
            factor = residual.get(pivot, Fraction(0))
            if factor == 0:
                continue
            for key, value in coeffs.items():
                residual[key] = residual.get(key, Fraction(0)) - factor * value
            residual = _nonzero(residual)
            constant += factor * rhs
            used.append((factor, row))
        return residual, constant, used

    def solve(self, free_assignment: Mapping[str, Fraction]) -> Dict[str, Fraction]:
        values = dict(free_assignment)
        for pivot, (coeffs, rhs, _source) in zip(self.pivot_cols, self.reduced):
            total = rhs
            for key, value in coeffs.items():
                if key != pivot:
                    total -= value * values.get(key, Fraction(0))
            values[pivot] = total
        return values


# --------------------------------------------------------------------------
class JointService:
    """Recovery service bound to a plan service and a scheme service."""

    def __init__(
        self,
        plans: PlanService,
        schemes: MeasurementSchemeService,
        connection: sqlite3.Connection,
    ) -> None:
        self.plans = plans
        self.schemes = schemes
        self._conn = connection
        self._conn.row_factory = sqlite3.Row
        self.readings = ReadingStore(connection)
        # Reuses the reduced linear system across repeated judgements of the
        # same reading version (e.g. several targets or several requested
        # output sets).  Keyed by the exact reading signature, so any new
        # effective reading -- even one arriving late -- invalidates it.
        self._system_cache: Dict[Tuple, _System] = {}
        self._create_schema()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS joint_proofs (
                proof_id TEXT PRIMARY KEY,
                record_id TEXT NOT NULL,
                scheme_id TEXT NOT NULL,
                scheme_digest TEXT NOT NULL,
                target_plan_id TEXT NOT NULL,
                target_plan_digest TEXT NOT NULL,
                requested_outputs TEXT NOT NULL,
                status TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_seq INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS joint_proof_readings (
                proof_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                record_id TEXT NOT NULL,
                scheme_id TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                value_num TEXT,
                value_den TEXT,
                PRIMARY KEY (proof_id, ordinal)
            );
            CREATE INDEX IF NOT EXISTS joint_proofs_record_idx
                ON joint_proofs (record_id, scheme_id);
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------ schemes
    def explain_scheme(
        self, scheme_id: str, channel_id: Optional[str] = None
    ) -> Dict[str, object]:
        return self.schemes.explain(scheme_id, channel_id)

    def _scheme(self, scheme_id: str) -> MeasurementScheme:
        try:
            return self.schemes.get(scheme_id)
        except UnknownChannelError as exc:
            raise UnknownSchemeError(str(exc)) from exc

    # ----------------------------------------------------------- readings
    def import_readings(
        self, scheme_id: str, raw_readings: Sequence[object]
    ) -> int:
        scheme = self._scheme(scheme_id)
        for index, raw in enumerate(raw_readings):
            if isinstance(raw, Mapping):
                declared = raw.get("scheme_id", scheme_id)
                if declared != scheme_id:
                    raise ReadingError(
                        f"reading {index} names scheme {declared!r}, but the "
                        f"import target is {scheme_id!r}"
                    )
                channel = raw.get("channel_id")
                if channel is not None and not scheme.has_channel(channel):
                    raise UnknownChannelError(
                        f"reading {index} refers to undeclared channel "
                        f"{channel!r}; scheme {scheme_id!r} declares "
                        f"{list(scheme.channel_map)}"
                    )
        normalised = []
        for raw in raw_readings:
            if isinstance(raw, Mapping) and "scheme_id" not in raw:
                item = dict(raw)
                item["scheme_id"] = scheme_id
                normalised.append(item)
            else:
                normalised.append(raw)
        return self.readings.import_readings(normalised)

    def retract(
        self,
        record_id: str,
        scheme_id: str,
        channel_id: str,
        revision: int,
    ) -> int:
        return self.import_readings(
            scheme_id,
            [
                {
                    "record_id": record_id,
                    "scheme_id": scheme_id,
                    "channel_id": channel_id,
                    "revision": revision,
                    "retract": True,
                }
            ],
        )

    def effective_readings(
        self, record_id: str, scheme_id: str
    ) -> Tuple[Reading, ...]:
        return self.readings.effective_for(record_id, scheme_id)

    def readings_for(
        self, scheme_id: str, record_id: Optional[str] = None
    ) -> List[Dict[str, object]]:
        scheme = self._scheme(scheme_id)
        if record_id is not None:
            return [
                {**_reading_ref(index, reading, scheme)}
                for index, reading in enumerate(
                    self.readings.effective_for(record_id, scheme_id)
                )
            ]
        result: List[Dict[str, object]] = []
        for rec in self.readings.record_ids():
            for index, reading in enumerate(
                self.readings.effective_for(rec, scheme_id)
            ):
                result.append(_reading_ref(index, reading, scheme))
        return result

    # ------------------------------------------------------------- solving
    def solve(
        self,
        record_id: str,
        scheme_id: str,
        target_plan_id: str,
        outputs: Optional[Iterable[str]] = None,
        proof_id: Optional[str] = None,
    ) -> Proof:
        scheme = self._scheme(scheme_id)
        try:
            plan = self.plans.get(target_plan_id)
        except Exception as exc:  # noqa: BLE001
            raise InvalidTargetError(str(exc)) from exc

        readings = self.readings.effective_for(record_id, scheme_id)
        if not readings:
            raise UnknownRecordError(
                f"no effective readings for record {record_id!r} under scheme "
                f"{scheme_id!r}"
            )
        requested = self._normalise_outputs(plan, outputs)

        universe = list(scheme.columns)
        for name in plan.inputs:
            if name not in universe:
                universe.append(name)
        if len(universe) > MAX_COLUMNS:
            raise InvalidTargetError(
                f"the local problem has {len(universe)} columns "
                f"(scheme plus plan inputs), limit is {MAX_COLUMNS}"
            )

        refs = [
            _reading_ref(index, reading, scheme)
            for index, reading in enumerate(readings)
        ]

        cache_key = (
            scheme_id,
            scheme.digest,
            tuple(universe),
            tuple(
                (reading.channel_id, reading.revision, reading.canonical_value())
                for reading in readings
            ),
        )
        system = self._system_cache.get(cache_key)
        if system is None:
            rows: List[Tuple[Dict[str, Fraction], Fraction]] = []
            for reading in readings:
                channel = scheme.channel(reading.channel_id)
                rows.append(
                    (
                        {
                            name: coefficient
                            for name, coefficient in channel.coefficients.items()
                        },
                        reading.value - channel.constant,  # type: ignore[operator]
                    )
                )
            system = _System(rows, universe)
            self._system_cache[cache_key] = system

        equations = [
            _equation_dict(reading, scheme.channel(reading.channel_id))
            for reading in readings
        ]
        snapshot = [
            {
                "record_id": reading.record_id,
                "scheme_id": reading.scheme_id,
                "channel_id": reading.channel_id,
                "revision": reading.revision,
                "value": _f(reading.value),  # type: ignore[arg-type]
            }
            for reading in readings
        ]

        version = {
            "record_id": record_id,
            "scheme_id": scheme_id,
            "scheme_digest": scheme.digest,
            "readings": snapshot,
        }

        if not system.consistent:
            conflict = self._conflict_payload(system, readings, scheme, refs)
            targets = [
                {
                    "output": output,
                    "status": "conflict",
                    "conflict": conflict,
                }
                for output in requested
            ]
            evaluation: Dict[str, object] = {
                "record_id": record_id,
                "scheme_id": scheme_id,
                "scheme_digest": scheme.digest,
                "target_plan_id": plan.plan_id,
                "target_plan_digest": plan.digest,
                "consistency": "conflict",
                "conflict": conflict,
                "equations": equations,
                "targets": targets,
            }
            status = "conflict"
        else:
            targets = []
            for output in requested:
                contract = plan.contract(output)
                target_vector = _nonzero(dict(contract.input_coefficients))
                targets.append(
                    self._judge_target(
                        system,
                        output,
                        contract,
                        target_vector,
                        readings,
                        scheme,
                        refs,
                    )
                )
            evaluation = {
                "record_id": record_id,
                "scheme_id": scheme_id,
                "scheme_digest": scheme.digest,
                "target_plan_id": plan.plan_id,
                "target_plan_digest": plan.digest,
                "consistency": "consistent",
                "conflict": None,
                "equations": equations,
                "targets": targets,
            }
            status = "consistent"

        payload: Dict[str, object] = {
            "version": version,
            "requested_outputs": list(requested),
            "status": status,
            "evaluation": evaluation,
            "scheme": scheme.as_explain_dict(),
            "target_plan": {
                "plan_id": plan.plan_id,
                "digest": plan.digest,
                "inputs": list(plan.inputs),
                "contracts": {
                    output: {
                        "constant": _f(plan.contract(output).constant),
                        "input_coefficients": {
                            name: _f(coefficient)
                            for name, coefficient in plan.contract(
                                output
                            ).input_coefficients.items()
                        },
                    }
                    for output in requested
                },
            },
        }
        return self._store(
            record_id=record_id,
            scheme_id=scheme_id,
            scheme_digest=scheme.digest,
            target_plan_id=plan.plan_id,
            target_plan_digest=plan.digest,
            requested=requested,
            status=status,
            payload=payload,
            readings=readings,
            proof_id=proof_id,
        )

    def _normalise_outputs(
        self, plan: Plan, outputs: Optional[Iterable[str]]
    ) -> Tuple[str, ...]:
        names = list(plan.outputs if outputs is None else outputs)
        result: List[str] = []
        for name in names:
            if not isinstance(name, str) or not name:
                raise InvalidTargetError(
                    f"requested output must be a non-empty string, got {name!r}"
                )
            if name not in plan.contracts:
                raise InvalidTargetError(
                    f"output {name!r} is not published by plan {plan.plan_id!r}; "
                    f"known outputs: {list(plan.outputs)}"
                )
            if name not in result:
                result.append(name)
        if not result:
            raise InvalidTargetError("at least one output must be requested")
        if len(result) > MAX_OUTPUTS:
            raise InvalidTargetError(
                f"requested {len(result)} outputs, limit is {MAX_OUTPUTS}"
            )
        return tuple(result)

    def _judge_target(
        self,
        system: _System,
        output: str,
        contract,
        target_vector: Mapping[str, Fraction],
        readings: Sequence[Reading],
        scheme: MeasurementScheme,
        refs: Sequence[Mapping[str, object]],
    ) -> Dict[str, object]:
        residual, constant, used = system.combine(target_vector)
        plan_constant = contract.constant
        if not residual:
            witness = self._witness(system, used, target_vector, constant, readings, scheme, refs)
            used_indices = set()
            for _factor, row in used:
                used_indices.update(row[2])
            return {
                "output": output,
                "status": "decided",
                "value": _f(plan_constant + constant),
                "linear_value": _f(constant),
                "plan_constant": _f(plan_constant),
                "target_coefficients": {
                    name: _f(coefficient)
                    for name, coefficient in target_vector.items()
                },
                "witness": witness,
                "used_readings": [refs[index] for index in sorted(used_indices)],
                "unused_readings": [
                    refs[index]
                    for index in range(len(refs))
                    if index not in used_indices
                ],
            }

        free_columns = system.free_columns()
        base_assignment = {column: Fraction(0) for column in free_columns}
        bump = next(
            column for column in free_columns if residual.get(column, Fraction(0))
        )
        alt_assignment = dict(base_assignment)
        alt_assignment[bump] = Fraction(1)

        base_values = system.solve(base_assignment)
        alt_values = system.solve(alt_assignment)

        def solution_payload(label: str, values: Mapping[str, Fraction], free):
            return {
                "label": label,
                "assignments": {
                    column: _f(values[column]) for column in system.columns
                },
                "free_columns": {
                    column: _f(free[column]) for column in free_columns
                },
                "target_value": _f(
                    self._dot(target_vector, values) + plan_constant
                ),
                "satisfies_readings": True,
                "readings_residual": {
                    reading.channel_id: _f(
                        self._reading_residual(reading, values, scheme)
                    )
                    for reading in readings
                },
            }

        return {
            "output": output,
            "status": "nonunique",
            "reason": (
                "the readings leave at least one column free and the target "
                "depends on it"
            ),
            "free_columns": free_columns,
            "pivot_columns": list(system.pivot_cols),
            "residual_dependence": {
                column: _f(coefficient)
                for column, coefficient in residual.items()
            },
            "solutions": [
                solution_payload("alpha", base_values, base_assignment),
                solution_payload("beta", alt_values, alt_assignment),
            ],
        }

    @staticmethod
    def _dot(vector: Mapping[str, Fraction], values: Mapping[str, Fraction]) -> Fraction:
        return sum(
            (coefficient * values.get(column, Fraction(0))
             for column, coefficient in vector.items()),
            Fraction(0),
        )

    @staticmethod
    def _reading_residual(
        reading: Reading, values: Mapping[str, Fraction], scheme: MeasurementScheme
    ) -> Fraction:
        channel = scheme.channel(reading.channel_id)
        total = channel.constant
        for column, coefficient in channel.coefficients.items():
            total += coefficient * values.get(column, Fraction(0))
        return total - reading.value  # type: ignore[operator]

    def _witness(
        self,
        system: _System,
        used: Sequence[Tuple[Fraction, Tuple]],
        target_vector: Mapping[str, Fraction],
        constant: Fraction,
        readings: Sequence[Reading],
        scheme: MeasurementScheme,
        refs: Sequence[Mapping[str, object]],
    ) -> Dict[str, object]:
        combine = []
        for factor, (coeffs, rhs, source) in used:
            combine.append(
                {
                    "factor": _f(factor),
                    "equation": {
                        "coefficients": {
                            column: _f(coefficient)
                            for column, coefficient in coeffs.items()
                        },
                        "rhs": _f(rhs),
                    },
                    "from": [
                        {
                            "reading": refs[index],
                            "multiplier": _f(multiplier),
                        }
                        for index, multiplier in sorted(source.items())
                    ],
                }
            )
        return {
            "target_coefficients": {
                name: _f(coefficient) for name, coefficient in target_vector.items()
            },
            "combine": combine,
            "sum_value": _f(constant),
            "check": (
                "sum(factor * equation.coefficients) equals the target "
                "coefficients and sum(factor * equation.rhs) equals sum_value; "
                "each equation is itself the stated combination of readings"
            ),
        }

    def _conflict_payload(
        self,
        system: _System,
        readings: Sequence[Reading],
        scheme: MeasurementScheme,
        refs: Sequence[Mapping[str, object]],
    ) -> Dict[str, object]:
        source = system.conflict_source or {}
        residual = system.conflict_residual or Fraction(0)
        terms = [
            {
                "reading": refs[index],
                "multiplier": _f(multiplier),
                "weighted_rhs": _f(
                    multiplier
                    * (readings[index].value - scheme.channel(readings[index].channel_id).constant)  # type: ignore[operator]
                ),
            }
            for index, multiplier in sorted(source.items())
        ]
        return {
            "reason": "the readings combine to 0 = a non-zero residual",
            "terms": terms,
            "zero_coefficients": {},
            "residual": _f(residual),
            "explanation": (
                "multiplying each listed reading by its multiplier cancels "
                "every column exactly, yet the right-hand sides sum to the "
                f"non-zero residual {_f(residual)}"
            ),
        }

    # --------------------------------------------------------- persistence
    def _next_seq(self) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(created_seq), 0) + 1 FROM joint_proofs"
        ).fetchone()
        return int(row[0])

    def _proof_exists(self, proof_id: str) -> bool:
        return (
            self._conn.execute(
                "SELECT 1 FROM joint_proofs WHERE proof_id = ?", (proof_id,)
            ).fetchone()
            is not None
        )

    def _next_proof_id(self) -> str:
        seq = self._next_seq()
        candidate = f"proof-{seq:06d}"
        while self._proof_exists(candidate):
            seq += 1
            candidate = f"proof-{seq:06d}"
        return candidate

    def _store(
        self,
        *,
        record_id: str,
        scheme_id: str,
        scheme_digest: str,
        target_plan_id: str,
        target_plan_digest: str,
        requested: Sequence[str],
        status: str,
        payload: Mapping[str, object],
        readings: Sequence[Reading],
        proof_id: Optional[str],
    ) -> Proof:
        if proof_id is None:
            proof_id = self._next_proof_id()
        elif self._proof_exists(proof_id):
            raise DuplicateProofError(
                f"proof {proof_id!r} already exists; proofs are immutable"
            )
        seq = self._next_seq()
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        with self._conn:
            self._conn.execute(
                "INSERT INTO joint_proofs (proof_id, record_id, scheme_id, "
                "scheme_digest, target_plan_id, target_plan_digest, "
                "requested_outputs, status, payload, created_seq) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    proof_id,
                    record_id,
                    scheme_id,
                    scheme_digest,
                    target_plan_id,
                    target_plan_digest,
                    json.dumps(list(requested)),
                    status,
                    encoded,
                    seq,
                ),
            )
            for ordinal, reading in enumerate(readings):
                value = reading.value
                self._conn.execute(
                    "INSERT INTO joint_proof_readings (proof_id, ordinal, "
                    "record_id, scheme_id, channel_id, revision, value_num, "
                    "value_den) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        proof_id,
                        ordinal,
                        reading.record_id,
                        reading.scheme_id,
                        reading.channel_id,
                        reading.revision,
                        None if value is None else str(value.numerator),
                        None if value is None else str(value.denominator),
                    ),
                )
        return self.reopen(proof_id)

    def reopen(self, proof_id: str) -> Proof:
        row = self._conn.execute(
            "SELECT * FROM joint_proofs WHERE proof_id = ?", (proof_id,)
        ).fetchone()
        if row is None:
            raise UnknownProofError(f"unknown proof {proof_id!r}")
        payload = json.loads(row["payload"])
        snapshot = tuple(payload["version"]["readings"])
        evaluation = payload["evaluation"]
        verification = self._verify_payload(payload)
        return Proof(
            proof_id=row["proof_id"],
            record_id=row["record_id"],
            scheme_id=row["scheme_id"],
            scheme_digest=row["scheme_digest"],
            target_plan_id=row["target_plan_id"],
            target_plan_digest=row["target_plan_digest"],
            requested_outputs=tuple(json.loads(row["requested_outputs"])),
            status=row["status"],
            evaluation=evaluation,
            readings_snapshot=snapshot,
            verification=verification,
            created_seq=row["created_seq"],
        )

    def read_proof(self, proof_id: str) -> Proof:
        return self.reopen(proof_id)

    def list_proofs(self) -> List[Dict[str, object]]:
        rows = self._conn.execute(
            "SELECT proof_id, record_id, scheme_id, scheme_digest, "
            "target_plan_id, target_plan_digest, requested_outputs, status, "
            "created_seq FROM joint_proofs ORDER BY created_seq"
        ).fetchall()
        return [
            {
                "proof_id": row["proof_id"],
                "record_id": row["record_id"],
                "scheme_id": row["scheme_id"],
                "scheme_digest": row["scheme_digest"],
                "target_plan_id": row["target_plan_id"],
                "target_plan_digest": row["target_plan_digest"],
                "requested_outputs": json.loads(row["requested_outputs"]),
                "status": row["status"],
                "created_seq": row["created_seq"],
            }
            for row in rows
        ]

    # --------------------------------------------------------- verification
    def verify(self, proof_id: str) -> Dict[str, object]:
        return dict(self.reopen(proof_id).verification)

    def _verify_payload(self, payload: Mapping[str, object]) -> Dict[str, object]:
        """Independently re-check a stored proof from its own snapshot."""
        failures: List[str] = []
        checks = 0
        evaluation = payload["evaluation"]  # type: ignore[index]
        contracts = payload["target_plan"]["contracts"]  # type: ignore[index]
        equations = {
            (item["record_id"], item["scheme_id"], item["channel_id"], item["revision"]): item
            for item in evaluation.get("equations", [])
        }

        def channel_terms(ref: Mapping[str, object]):
            coeffs = {
                name: as_fraction(value)
                for name, value in ref["coefficients"].items()  # type: ignore[union-attr]
            }
            constant = as_fraction(ref["constant"])  # type: ignore[arg-type]
            value = as_fraction(ref["value"])  # type: ignore[arg-type]
            return coeffs, constant, value

        for target in evaluation.get("targets", []):
            output = target["output"]
            contract = contracts.get(output)  # type: ignore[union-attr]
            if contract is None:
                failures.append(f"missing target contract for {output}")
                continue
            expected_vector = {
                name: as_fraction(value)
                for name, value in contract["input_coefficients"].items()
            }
            expected_constant = as_fraction(contract["constant"])
            status = target["status"]
            if status == "decided":
                checks += 1
                total = Fraction(0)
                vector: Dict[str, Fraction] = {}
                for entry in target["witness"]["combine"]:
                    factor = as_fraction(entry["factor"])
                    for source in entry["from"]:
                        ref = source["reading"]
                        multiplier = factor * as_fraction(source["multiplier"])
                        coeffs, constant, value = channel_terms(ref)
                        total += multiplier * (value - constant)
                        for name, coefficient in coeffs.items():
                            vector[name] = vector.get(name, Fraction(0)) + multiplier * coefficient
                vector = _nonzero(vector)
                if vector != _nonzero(expected_vector):
                    failures.append(
                        f"{output}: witness coefficients do not reproduce the plan"
                    )
                expected_value = as_fraction(target["value"]) - expected_constant
                if total != as_fraction(target["linear_value"]):
                    failures.append(f"{output}: witness value mismatch")
                if expected_value != as_fraction(target["linear_value"]):
                    failures.append(f"{output}: stored value inconsistent")
            elif status == "nonunique":
                checks += 1
                seen = set()
                target_failures: List[str] = []
                for solution in target["solutions"]:
                    values = {
                        name: as_fraction(value)
                        for name, value in solution["assignments"].items()
                    }
                    for key, equation in equations.items():
                        coeffs, constant, value = channel_terms(equation)
                        lhs = constant + sum(
                            (coefficient * values.get(name, Fraction(0))
                             for name, coefficient in coeffs.items()),
                            Fraction(0),
                        )
                        if lhs != value:
                            target_failures.append(
                                f"{output}: solution does not satisfy reading {key}"
                            )
                    recomputed = expected_constant + sum(
                        (coefficient * values.get(name, Fraction(0))
                         for name, coefficient in expected_vector.items()),
                        Fraction(0),
                    )
                    if recomputed != as_fraction(solution["target_value"]):
                        target_failures.append(
                            f"{output}: solution target value does not match "
                            "the plan contract"
                        )
                    seen.add(as_fraction(solution["target_value"]))
                failures.extend(dict.fromkeys(target_failures))
                if len(seen) < 2:
                    failures.append(f"{output}: solutions do not differ")
            elif status == "conflict":
                checks += 1
                residual = Fraction(0)
                vector = {}
                for term in target["conflict"]["terms"]:
                    ref = term["reading"]
                    multiplier = as_fraction(term["multiplier"])
                    coeffs, constant, value = channel_terms(ref)
                    residual += multiplier * (value - constant)
                    for name, coefficient in coeffs.items():
                        vector[name] = vector.get(name, Fraction(0)) + multiplier * coefficient
                if _nonzero(vector):
                    failures.append(
                        f"{output}: conflict combination does not cancel columns"
                    )
                if residual == 0:
                    failures.append(f"{output}: conflict residual is zero")
        return {
            "ok": not failures,
            "checks": checks,
            "failures": failures,
            "scheme_digest": payload["version"]["scheme_digest"],  # type: ignore[index]
            "target_plan_digest": evaluation["target_plan_digest"],
        }


def open_joint(
    plans: PlanService,
    schemes: MeasurementSchemeService,
    connection: Optional[sqlite3.Connection] = None,
    path: str = ":memory:",
) -> JointService:
    if connection is None:
        connection = sqlite3.connect(path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
    return JointService(plans, schemes, connection)
