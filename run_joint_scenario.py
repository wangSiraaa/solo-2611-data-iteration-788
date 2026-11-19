#!/usr/bin/env python3.11
"""Demonstrate joint-reading recovery end to end.

Run with::

    python3.11 run_joint_scenario.py

The bench exposes named linear channels ``M = a + b`` and ``D = a - b + 1``
(and later an independent ``C = 2a + 2b``).  The plan's targets are
``shifted = a + b + 2``, ``a`` and ``b``.  The script shows, per record:

1. ``a + b = 7`` alone: ``shifted = 9`` is decided; ``a`` is non-unique and two
   concrete explanations are produced.
2. adding ``a - b + 1 = 2``: ``a = 4``, ``b = 3``.
3. an independent third channel reporting ``2a + 2b = 15`` conflicts.
4. withdrawing that channel restores identifiability.
5. reopening each proof after a fresh connection verifies it.

Nothing here is guessed: every value is exact and every conclusion is
independent of the other targets.
"""

from __future__ import annotations

import sys

from exact_ledger import (
    MeasurementSchemeService,
    PlanService,
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
        {"channel_id": "C", "terms": [
            {"ref": "a", "coefficient": "2"},
            {"ref": "b", "coefficient": "2"},
        ]},
    ],
}

TARGET_PLAN = {
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


def show(proof, label):
    print(f"\n-- {label}")
    print(f"   proof={proof.proof_id} status={proof.status} "
          f"readings={len(proof.readings_snapshot)} "
          f"verify={'ok' if proof.verification['ok'] else 'FAILED'}")
    for target in proof.evaluation["targets"]:
        status = target["status"]
        if status == "decided":
            print(f"   {target['output']}: decided = {target['value']}")
            sources = sorted({r["channel_id"] for r in target["used_readings"]})
            unused = sorted({r["channel_id"] for r in target["unused_readings"]})
            print(f"      witness from {sources}"
                  + (f"; unused redundant {unused}" if unused else ""))
        elif status == "nonunique":
            pairs = ", ".join(
                "(" + ", ".join(f"{k}={v}" for k, v in s["assignments"].items())
                + f") -> {s['target_value']}"
                for s in target["solutions"]
            )
            print(f"   {target['output']}: nonunique; {pairs}")
        else:
            terms = ", ".join(
                f"{t['multiplier']}*{t['reading']['channel_id']}"
                for t in target["conflict"]["terms"]
            )
            print(f"   {target['output']}: conflict ({terms}; "
                  f"residual {target['conflict']['residual']})")


def main(argv):
    plans = PlanService()
    schemes = MeasurementSchemeService()
    schemes.publish(SCHEME)
    plans.publish(TARGET_PLAN)

    conn = open_connection()
    joint = open_joint(plans, schemes, connection=conn)

    print("== scheme ==")
    for channel in SCHEME["channels"]:
        terms = " + ".join(
            f"{t['coefficient']}*{t['ref']}" for t in channel["terms"]
        )
        constant = channel.get("constant", "0")
        print(f"  {channel['channel_id']} = {terms} + {constant}")

    print("\n== 1. a + b = 7 only ==")
    joint.import_readings("bench", [
        {"record_id": "r1", "channel_id": "M", "revision": 1, "value": "7"},
    ])
    proof = joint.solve("r1", "bench", "targets", ["shifted", "a"],
                        proof_id="sum-only")
    show(proof, "only the sum is known")

    print("\n== 2. add a - b + 1 = 2 ==")
    joint.import_readings("bench", [
        {"record_id": "r1", "channel_id": "D", "revision": 1, "value": "2"},
    ])
    proof = joint.solve("r1", "bench", "targets", ["a", "b", "shifted"],
                        proof_id="fully-determined")
    show(proof, "two independent channels")

    print("\n== 3. independent third channel 2a + 2b = 15 ==")
    joint.import_readings("bench", [
        {"record_id": "r1", "channel_id": "C", "revision": 1, "value": "15"},
    ])
    proof = joint.solve("r1", "bench", "targets", ["a", "shifted"],
                        proof_id="conflict")
    show(proof, "contradictory channels")

    print("\n== 4. withdraw channel C ==")
    joint.retract("r1", "bench", "C", 2)
    proof = joint.solve("r1", "bench", "targets", ["a", "b", "shifted"],
                        proof_id="restored")
    show(proof, "after withdrawal")

    print("\n== 5. reopen every proof on a fresh connection ==")
    reopened = open_joint(plans, schemes, connection=conn)
    for entry in reopened.list_proofs():
        proof = reopened.reopen(entry["proof_id"])
        print(f"   {proof.proof_id}: status={proof.status} "
              f"readings={len(proof.readings_snapshot)} "
              f"verify={'ok' if proof.verification['ok'] else 'FAILED'}")
    # the old non-unique proof still explains a with its original single reading
    old = reopened.reopen("sum-only")
    a_target = old.target("a")
    print("   sum-only still nonunique for a:",
          [s["target_value"] for s in a_target["solutions"]])
    print("   sum-only shifted =", reopened.reopen("sum-only").target("shifted")["value"])


if __name__ == "__main__":
    main(sys.argv)
