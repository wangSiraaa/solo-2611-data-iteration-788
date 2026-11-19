"""Tests for measurement schemes, joint readings and identifiability proofs."""

from __future__ import annotations

import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

from exact_ledger import (
    DuplicateChannelError,
    DuplicateProofError,
    InvalidSchemeError,
    MeasurementSchemeService,
    PlanService,
    ReadingConflictError,
    UnknownChannelError,
    UnknownColumnError,
    UnknownProofError,
    UnknownRecordError,
    UnknownSchemeError,
    as_fraction,
    open_connection,
    open_joint,
    to_str,
)

SCHEME = {
    "scheme_id": "bench",
    "columns": ["a", "b"],
    "channels": [
        {"channel_id": "M", "terms": [
            {"ref": "a", "coefficient": "1"},
            {"ref": "b", "coefficient": "1"},
        ]},
        {"channel_id": "D", "constant": "1", "terms": [
            {"ref": "a", "coefficient": "1"},
            {"ref": "b", "coefficient": "-1"},
        ]},
    ],
}

TARGET = {
    "plan_id": "targets",
    "inputs": ["a", "b"],
    "nodes": [
        {"node_id": "shifted", "constant": "2", "terms": [
            {"ref": "a", "coefficient": "1"},
            {"ref": "b", "coefficient": "1"},
        ]},
    ],
    "outputs": ["shifted", "a", "b"],
}


def _target_statuses(proof):
    return {t["output"]: t["status"] for t in proof.evaluation["targets"]}


class Fixture(unittest.TestCase):
    def setUp(self):
        self.plans = PlanService()
        self.schemes = MeasurementSchemeService()
        self.schemes.publish(SCHEME)
        self.plans.publish(TARGET)
        self.joint = open_joint(self.plans, self.schemes)

    def read(self, channel, value, record="r1", revision=1, scheme="bench"):
        return self.joint.import_readings(
            scheme,
            [{"record_id": record, "channel_id": channel,
              "revision": revision, "value": value}],
        )


class SchemeTests(unittest.TestCase):
    def setUp(self):
        self.service = MeasurementSchemeService()

    def test_scheme_products_are_exact_and_reduced(self):
        scheme = self.service.publish({
            "scheme_id": "s",
            "columns": ["a", "b"],
            "channels": [
                {"channel_id": "z", "constant": "1/2", "terms": [
                    {"ref": "a", "coefficient": "1"},
                    {"ref": "b", "coefficient": "-1"},
                    {"ref": "b", "coefficient": "1"},
                ]},
            ],
        })
        channel = scheme.channel("z")
        self.assertEqual(channel.constant, Fraction(1, 2))
        self.assertEqual(channel.coefficients, {"a": Fraction(1)})

    def test_scheme_is_immutable(self):
        first = self.service.publish(SCHEME)
        again = self.service.publish(SCHEME)
        self.assertIs(first, again)
        changed = dict(SCHEME)
        changed["channels"] = [
            {"channel_id": "M", "terms": [{"ref": "a", "coefficient": "2"}]}
        ]
        with self.assertRaises(InvalidSchemeError):
            self.service.publish(changed)

    def test_rejects_unknown_column_and_duplicate_channel(self):
        with self.assertRaises(UnknownColumnError):
            self.service.publish({
                "scheme_id": "bad", "columns": ["a"],
                "channels": [{"channel_id": "c", "terms": [
                    {"ref": "ghost", "coefficient": "1"}]}],
            })
        with self.assertRaises(DuplicateChannelError):
            self.service.publish({
                "scheme_id": "dup", "columns": ["a"],
                "channels": [
                    {"channel_id": "c", "terms": [{"ref": "a", "coefficient": "1"}]},
                    {"channel_id": "c", "terms": [{"ref": "a", "coefficient": "2"}]},
                ],
            })

    def test_unknown_scheme_lookup(self):
        with self.assertRaises(UnknownChannelError):
            self.service.get("nope")


class ReadingImportTests(Fixture):
    def test_all_or_nothing_on_conflicting_identity(self):
        self.read("M", "7")
        with self.assertRaises(ReadingConflictError):
            self.joint.import_readings("bench", [
                {"record_id": "r1", "channel_id": "M", "revision": 1, "value": "8"},
                {"record_id": "r1", "channel_id": "D", "revision": 1, "value": "2"},
            ])
        # the second, conflict-free row of the rejected batch did not land
        self.assertIsNone(self.joint.readings.get_revision("r1", "bench", "D", 1))

    def test_equivalent_fraction_is_idempotent(self):
        self.assertEqual(self.read("M", "7"), 1)
        self.assertEqual(self.read("M", "14/2"), 0)
        self.assertEqual(self.read("M", "7.0"), 0)
        revisions = self.joint.readings.all_revisions("r1", "bench", "M")
        self.assertEqual(len(revisions), 1)

    def test_late_lower_revision_is_stored_but_ignored(self):
        self.read("M", "7", revision=1)
        self.read("M", "9", revision=2)
        # a late revision 1 packet that *differs* is rejected on identity
        with self.assertRaises(ReadingConflictError):
            self.read("M", "5", revision=1)
        # re-sending the old value is an idempotent replay, still ignored
        self.assertEqual(self.read("M", "7", revision=1), 0)
        effective = self.joint.effective_readings("r1", "bench")
        self.assertEqual([(r.channel_id, r.revision, r.value) for r in effective],
                         [("M", 2, Fraction(9))])
        # both revisions retained
        self.assertEqual(
            [(r.revision, r.value)
             for r in self.joint.readings.all_revisions("r1", "bench", "M")],
            [(1, Fraction(7)), (2, Fraction(9))],
        )

    def test_late_arriving_new_channel_counts(self):
        self.read("M", "7", revision=5)
        self.read("D", "2", revision=1)  # lower rev, but first time this channel
        self.assertEqual(
            _target_statuses(self.joint.solve("r1", "bench", "targets", ["a"])),
            {"a": "decided"},
        )

    def test_retract_and_restore(self):
        self.read("M", "7")
        self.read("D", "2")
        self.joint.retract("r1", "bench", "D", 2)
        self.assertEqual(
            [r.channel_id for r in self.joint.effective_readings("r1", "bench")],
            ["M"],
        )
        self.read("D", "2", revision=3)
        self.assertEqual(
            _target_statuses(self.joint.solve("r1", "bench", "targets", ["a"])),
            {"a": "decided"},
        )

    def test_import_validates_declared_channel_and_scheme(self):
        with self.assertRaises(UnknownChannelError):
            self.read("zzz", "1")
        with self.assertRaises(UnknownSchemeError):
            self.joint.import_readings(
                "nope", [{"record_id": "r1", "channel_id": "M",
                          "revision": 1, "value": "1"}]
            )


class IdentifiabilityTests(Fixture):
    def test_sum_target_is_identifiable_while_components_are_not(self):
        self.read("M", "7")
        proof = self.joint.solve("r1", "bench", "targets", ["shifted", "a"])
        statuses = _target_statuses(proof)
        self.assertEqual(statuses["shifted"], "decided")
        self.assertEqual(statuses["a"], "nonunique")
        shifted = proof.target("shifted")
        self.assertEqual(as_fraction(shifted["value"]), Fraction(9))
        # irrelevant target did not block the identifiable one
        self.assertTrue(proof.verification["ok"])

    def test_nonunique_gives_two_concrete_solutions(self):
        self.read("M", "7")
        proof = self.joint.solve("r1", "bench", "targets", ["a"])
        target = proof.target("a")
        solutions = target["solutions"]
        self.assertEqual(len(solutions), 2)
        values = [as_fraction(s["target_value"]) for s in solutions]
        self.assertNotEqual(values[0], values[1])
        assignments = {
            tuple(sorted((k, as_fraction(v)) for k, v in s["assignments"].items()))
            for s in solutions
        }
        # both concrete explanations satisfy a + b = 7
        for assignment in assignments:
            values = dict(assignment)
            self.assertEqual(values["a"] + values["b"], Fraction(7))
        # the prose's illustrative pair also satisfies the same reading
        for a_value, b_value in ((Fraction(0), Fraction(7)),
                                 (Fraction(1), Fraction(6))):
            self.assertEqual(a_value + b_value, Fraction(7))
        for solution in solutions:
            residual = {
                k: as_fraction(v)
                for k, v in solution["readings_residual"].items()
            }
            self.assertEqual(residual.get("M", Fraction(0)), Fraction(0))

    def test_two_channels_make_components_decided(self):
        self.read("M", "7")
        self.read("D", "2")  # a - b + 1 = 2
        proof = self.joint.solve("r1", "bench", "targets", ["a", "b", "shifted"])
        self.assertEqual(_target_statuses(proof),
                         {"a": "decided", "b": "decided", "shifted": "decided"})
        self.assertEqual(as_fraction(proof.target("a")["value"]), Fraction(4))
        self.assertEqual(as_fraction(proof.target("b")["value"]), Fraction(3))
        self.assertEqual(as_fraction(proof.target("shifted")["value"]), Fraction(9))

    def test_witness_reproduces_target_coefficients(self):
        self.read("M", "7")
        self.read("D", "2")
        proof = self.joint.solve("r1", "bench", "targets", ["a"])
        witness = proof.target("a")["witness"]
        coefficients = {}
        total = Fraction(0)
        for entry in witness["combine"]:
            factor = as_fraction(entry["factor"])
            for source in entry["from"]:
                multiplier = factor * as_fraction(source["multiplier"])
                ref = source["reading"]
                total += multiplier * (
                    as_fraction(ref["value"]) - as_fraction(ref["constant"])
                )
                for name, coefficient in ref["coefficients"].items():
                    coefficients[name] = coefficients.get(name, Fraction(0)) + (
                        multiplier * as_fraction(coefficient)
                    )
        self.assertEqual(
            {k: v for k, v in coefficients.items() if v}, {"a": Fraction(1)}
        )
        self.assertEqual(total, Fraction(4))
        # independently substitute back through the actual channels
        self.assertEqual(
            Fraction(1, 2) * (Fraction(7) - 0) + Fraction(1, 2) * (Fraction(2) - 1),
            Fraction(4),
        )

    def test_redundant_reading_still_decides_and_is_available(self):
        # A third channel R = a + b duplicates M; the system is redundant but
        # consistent, and the extra reading must not change the conclusion.
        self.schemes.publish({
            "scheme_id": "redundant",
            "columns": ["a", "b"],
            "channels": SCHEME["channels"] + [
                {"channel_id": "R", "terms": [
                    {"ref": "a", "coefficient": "1"},
                    {"ref": "b", "coefficient": "1"},
                ]},
            ],
        })
        self.joint.import_readings("redundant", [
            {"record_id": "r1", "channel_id": "M", "revision": 1, "value": "7"},
            {"record_id": "r1", "channel_id": "D", "revision": 1, "value": "2"},
            {"record_id": "r1", "channel_id": "R", "revision": 1, "value": "7"},
        ])
        proof = self.joint.solve("r1", "redundant", "targets", ["a", "b"])
        self.assertEqual(proof.status, "consistent")
        self.assertEqual(_target_statuses(proof), {"a": "decided", "b": "decided"})
        self.assertEqual(as_fraction(proof.target("a")["value"]), Fraction(4))
        self.assertEqual(as_fraction(proof.target("b")["value"]), Fraction(3))
        # the redundant reading is preserved in the proof's snapshot
        self.assertEqual(len(proof.readings_snapshot), 3)
        self.assertTrue(proof.verification["ok"])

    def test_conflicting_third_channel_is_reported(self):
        self.schemes.publish({
            "scheme_id": "bench3",
            "columns": ["a", "b"],
            "channels": SCHEME["channels"] + [
                {"channel_id": "C", "terms": [
                    {"ref": "a", "coefficient": "2"},
                    {"ref": "b", "coefficient": "2"},
                ]},
            ],
        })
        self.joint.import_readings("bench3", [
            {"record_id": "r1", "channel_id": "M", "revision": 1, "value": "7"},
            {"record_id": "r1", "channel_id": "D", "revision": 1, "value": "2"},
            {"record_id": "r1", "channel_id": "C", "revision": 1, "value": "15"},
        ])
        proof = self.joint.solve("r1", "bench3", "targets", ["a", "shifted"])
        self.assertEqual(proof.status, "conflict")
        self.assertEqual(_target_statuses(proof),
                         {"a": "conflict", "shifted": "conflict"})
        conflict = proof.target("a")["conflict"]
        # equations cancel every column but leave a non-zero residual
        vector = {}
        residual = Fraction(0)
        for term in conflict["terms"]:
            multiplier = as_fraction(term["multiplier"])
            ref = term["reading"]
            residual += multiplier * (
                as_fraction(ref["value"]) - as_fraction(ref["constant"])
            )
            for name, coefficient in ref["coefficients"].items():
                vector[name] = vector.get(name, Fraction(0)) + (
                    multiplier * as_fraction(coefficient)
                )
        self.assertEqual({k: v for k, v in vector.items() if v}, {})
        self.assertNotEqual(residual, 0)
        self.assertTrue(proof.verification["ok"])

        # withdrawing the offending channel restores identifiability
        self.joint.retract("r1", "bench3", "C", 2)
        restored = self.joint.solve("r1", "bench3", "targets", ["a", "shifted"])
        self.assertEqual(restored.status, "consistent")
        self.assertEqual(_target_statuses(restored),
                         {"a": "decided", "shifted": "decided"})
        self.assertEqual(as_fraction(restored.target("a")["value"]), Fraction(4))

    def test_direct_column_reading_is_a_sourced_equation(self):
        self.schemes.publish({
            "scheme_id": "direct",
            "columns": ["a", "b"],
            "channels": [
                {"channel_id": "a", "terms": [{"ref": "a", "coefficient": "1"}]},
            ],
        })
        self.joint.import_readings("direct", [
            {"record_id": "r1", "channel_id": "a", "revision": 4, "value": "4"},
        ])
        proof = self.joint.solve("r1", "direct", "targets", ["a", "b"])
        equations = proof.evaluation["equations"]
        self.assertEqual(len(equations), 1)
        self.assertEqual(equations[0]["direct_column"], "a")
        self.assertEqual(equations[0]["value"], "4")
        self.assertEqual(equations[0]["revision"], 4)
        self.assertEqual(_target_statuses(proof), {"a": "decided", "b": "nonunique"})

    def test_constant_target_needs_no_readings_dependence(self):
        self.schemes.publish({
            "scheme_id": "const",
            "columns": ["a"],
            "channels": [{"channel_id": "a", "terms": [{"ref": "a", "coefficient": "1"}]}],
        })
        self.joint.import_readings("const", [
            {"record_id": "r1", "channel_id": "a", "revision": 1, "value": "4"},
        ])
        # a target plan with a constant-only output
        self.plans.publish({
            "plan_id": "const-plan", "inputs": ["a"],
            "nodes": [{"node_id": "k", "constant": "5", "terms": []}],
            "outputs": ["k"],
        })
        proof = self.joint.solve("r1", "const", "const-plan", ["k"])
        self.assertEqual(as_fraction(proof.target("k")["value"]), Fraction(5))

    def test_unknown_record_and_target(self):
        with self.assertRaises(UnknownRecordError):
            self.joint.solve("nobody", "bench", "targets", ["a"])
        self.read("M", "7")
        from exact_ledger import InvalidTargetError
        with self.assertRaises(InvalidTargetError):
            self.joint.solve("r1", "bench", "targets", ["missing"])

    def test_column_limit_enforced(self):
        from exact_ledger import InvalidTargetError
        # scheme already holds the maximum 8 columns; a target plan that
        # introduces a ninth input pushes the local problem over the bound.
        many = list("abcdefgh")
        self.schemes.publish({
            "scheme_id": "wide",
            "columns": many,
            "channels": [{"channel_id": "c", "terms": [
                {"ref": "a", "coefficient": "1"}]}],
        })
        self.plans.publish({
            "plan_id": "wide-plan", "inputs": ["a", "ninth"],
            "nodes": [], "outputs": ["a"],
        })
        self.joint.import_readings("wide", [
            {"record_id": "r1", "channel_id": "c", "revision": 1, "value": "1"},
        ])
        with self.assertRaises(InvalidTargetError):
            self.joint.solve("r1", "wide", "wide-plan", ["a"])

    def test_repeated_solves_share_reduced_system(self):
        self.read("M", "7")
        first = self.joint.solve("r1", "bench", "targets", ["a"])
        cached = len(self.joint._system_cache)
        second = self.joint.solve("r1", "bench", "targets", ["b"])
        self.assertEqual(len(self.joint._system_cache), cached)
        self.assertEqual(first.status, second.status)
        # a new effective reading invalidates the cache key
        self.read("D", "2")
        self.joint.solve("r1", "bench", "targets", ["a"])
        self.assertEqual(len(self.joint._system_cache), cached + 1)


class ProofPersistenceTests(Fixture):
    def test_proof_survives_reopen_and_old_proofs_are_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "joint.sqlite")
            plans = PlanService()
            plans.publish(TARGET)
            schemes = MeasurementSchemeService()
            schemes.publish(SCHEME)
            first = open_joint(plans, schemes, path=path)
            first.import_readings("bench", [
                {"record_id": "r1", "channel_id": "M", "revision": 1, "value": "7"},
            ])
            proof = first.solve("r1", "bench", "targets", ["a", "shifted"],
                                proof_id="under-determined")
            self.assertEqual(proof.status, "consistent")
            self.assertEqual(_target_statuses(proof)["a"], "nonunique")

            # more readings arrive; a new proof changes conclusions
            first.import_readings("bench", [
                {"record_id": "r1", "channel_id": "D", "revision": 1, "value": "2"},
            ])
            later = first.solve("r1", "bench", "targets", ["a"],
                                proof_id="determined")
            self.assertEqual(as_fraction(later.target("a")["value"]), Fraction(4))

            # a fresh connection sees both, each pinned to its own reading set
            reopened = open_joint(plans, schemes, path=path)
            old = reopened.reopen("under-determined")
            self.assertEqual(_target_statuses(old)["a"], "nonunique")
            self.assertEqual(len(old.readings_snapshot), 1)
            self.assertEqual(
                [r["revision"] for r in old.readings_snapshot], [1]
            )
            self.assertTrue(old.verification["ok"])
            new = reopened.reopen("determined")
            self.assertEqual(as_fraction(new.target("a")["value"]), Fraction(4))
            self.assertEqual(len(new.readings_snapshot), 2)
            self.assertTrue(new.verification["ok"])

    def test_duplicate_proof_id_rejected(self):
        self.read("M", "7")
        self.joint.solve("r1", "bench", "targets", ["a"], proof_id="p")
        with self.assertRaises(DuplicateProofError):
            self.joint.solve("r1", "bench", "targets", ["a"], proof_id="p")
        with self.assertRaises(UnknownProofError):
            self.joint.reopen("ghost")

    def test_scheme_and_plan_digests_recorded(self):
        self.read("M", "7")
        proof = self.joint.solve("r1", "bench", "targets", ["shifted"])
        self.assertEqual(proof.scheme_digest, self.schemes.get("bench").digest)
        self.assertEqual(proof.target_plan_digest, self.plans.get("targets").digest)


class MultipleRecordTests(Fixture):
    def test_scheme_reused_across_records(self):
        self.read("M", "7", record="r1")
        self.read("M", "10", record="r2")
        proof1 = self.joint.solve("r1", "bench", "targets", ["shifted"])
        proof2 = self.joint.solve("r2", "bench", "targets", ["shifted"])
        self.assertEqual(as_fraction(proof1.target("shifted")["value"]), Fraction(9))
        self.assertEqual(as_fraction(proof2.target("shifted")["value"]), Fraction(12))
        # only r1's readings are in r1's proof
        self.assertEqual(
            {r["record_id"] for r in proof1.readings_snapshot}, {"r1"}
        )


if __name__ == "__main__":
    unittest.main()
