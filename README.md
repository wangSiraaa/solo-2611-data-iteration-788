# exact_ledger — exact linear plans + grouped fact replay

Python 3.11, standard library only (SQLite is used for persistence).  Two
deliberately separate components:

| Component | Module | Owns |
| --- | --- | --- |
| Plan service | `exact_ledger/plan.py` | Plans, DAG validation, exact algebra, published per-output contracts. **Knows nothing about facts.** |
| Replay warehouse | `exact_ledger/replay.py`, `exact_ledger/facts.py` | Fact versions, groups, import rules, eligibility, immutable results and provenance. **Consumes the published plan products.** |
| Measurement schemes | `exact_ledger/scheme.py` | Immutable named linear channels (`M = a + b`, `D = a - b + 1`). Knows nothing about readings. |
| Joint readings | `exact_ledger/readings.py`, `exact_ledger/joint.py` | Versioned per-record readings, per-output identifiability proofs, conflict equations, persistence. **Consumes published schemes and plan contracts.** |

`exact_ledger/rational.py` is the exact-rational layer: ints, `Fraction`, and
string literals `"3"`, `"-3/4"`, `"0.125"`.  No float ever enters evaluation.

## Plan service

A plan has named inputs and nodes; each node is

```
node = constant + Σ coefficientᵢ · refᵢ          (refᵢ = input name or node id)
```

Terms may repeat, coefficients may be zero or negative, and a node may be
referenced from many places.  The graph must be acyclic; unknown refs and
empty ids are rejected.  Merged coefficients that reduce to zero are
**eliminated** from every published product, and names are compared literally
(`"a"`, `"aa"`, `"a_a"` are distinct).

```python
from exact_ledger import PlanService
plans = PlanService()
p1 = plans.publish(plan_definition)          # immutable; same content => same object
p1.contract("z").input_coefficients          # {'a': Fraction(1)}  (b cancelled)
plans.explain("p1", "z")                     # contract + raw node products
plans.explain_coefficient("p1", "z", "b")    # why b's final coefficient is 0
p2 = plans.publish_replacement("p2", "p1", replacement_node)  # copy + replace
```

Published plans are immutable: re-publishing a `plan_id` with different
content is rejected.  Each plan carries a content `digest`.

`explain_coefficient` runs one exact reverse-topological pass over the shared
DAG (never path enumeration).  For `z = u − b`, `u = a + b` it reports
`b`'s final coefficient `0` as `cancelled`, with the signed edge contributions
`u->b: +1`, `z->b: −1` that sum to zero — so a shared source is not double
counted and a zero dependency is never mistaken for a value change.

## Replay warehouse

Facts are keyed by `record_id`, with a positive `revision`, a `group_id`, and a
**subset** of declared inputs.  Missing input = **unavailable**, never zero.

* highest revision wins; earlier revisions are retained but ignored;
* same revision + same canonical content = idempotent replay;
* same revision + different content = **whole batch rejected**, nothing written;
* `group_id` may change only in a *higher, explicitly `complete`* fact
  (an explicit complete fact must cover all declared inputs);
* incomplete facts are legal to store — never coerced to zero, never rejected.

```python
from exact_ledger import open_warehouse
warehouse = open_warehouse(plans)                   # in-memory SQLite; or path=...
warehouse.import_facts("p1", facts)                 # atomic batch
report = warehouse.inspect("p1", ["z", "w"])        # eligibility for the requested set
result = warehouse.publish("p1", ["z"], result_id="initial")
result.total("g", "z")                              # Fraction(4)
result.trace("r1")                                  # revision -> nonzero contributions
warehouse.reopen("initial")                         # immutable, survives reconnect
```

### Eligibility is per requested output set

For every record and every **requested** output, every input in that output's
**reduced, non-zero** dependency map must be present.  Therefore:

* an input with coefficient `0` for all requested outputs may be absent;
* a constant-only output is decidable with no inputs at all;
* if one record lacks a needed input for one requested output, the **whole
  request is rejected** (no half group), and a `rejected` result records the
  `record_id` / `input` / `output` / coefficient reasons;
* plan legality is never treated as “every output has a value”.

### Results are immutable

Every publish attempt gets its own identity and records the plan digest and
fact snapshot.  Old results keep their old plan and old facts; a later
rejection never overwrites an earlier published result.  `recompute` is a full
recompute; `incremental_recompute` reuses per-record values only where the
fact snapshot and reduced contract are unchanged, but always rebuilds totals
from the current record set so constants and group migrations are not missed.

Evaluation is a sparse dot product over the reduced dependency map — the
warehouse never “propagates UNKNOWN” through the original expression, so
`(a+b)−b` with `b` missing is correctly a non-dependency, not “uncertain”.

## Joint readings and identifiability proofs

Two channels of one bench may only be observed jointly (`a + b = 7`) without
either being measured separately.  A **measurement scheme** publishes named
channels as exact rational linear readings with an optional constant:

```python
from exact_ledger import MeasurementSchemeService, PlanService, open_joint
schemes = MeasurementSchemeService()
schemes.publish({
    "scheme_id": "bench",
    "columns": ["a", "b"],
    "channels": [
        {"channel_id": "M", "terms": [{"ref": "a", "coefficient": "1"},
                                      {"ref": "b", "coefficient": "1"}]},
        {"channel_id": "D", "constant": "1",
         "terms": [{"ref": "a", "coefficient": "1"},
                   {"ref": "b", "coefficient": "-1"}]},
    ],
})
```

Schemes are published independently and are immutable per `scheme_id`; a
channel of a version can never be rewritten in place (publish a new scheme
version instead).  A scheme may be reused by many records.

A **reading** is identified by `(record_id, scheme_id, channel_id, revision)`:

```python
joint = open_joint(plans, schemes, path="joint.sqlite")   # SQLite-backed
joint.import_readings("bench", [
    {"record_id": "r1", "channel_id": "M", "revision": 1, "value": "7"},
])
proof = joint.solve("r1", "bench", "targets", ["shifted", "a"])
proof.target("shifted")["value"]      # "9"  (decided)
proof.target("a")["status"]           # "nonunique", with two solutions
joint.retract("r1", "bench", "D", 2)  # withdraw a channel in a higher revision
```

Import rules:

* highest revision wins; a **late** lower revision is stored but ignored;
* same identity with the same canonical value (`7`, `14/2`, `7.0`) is an
  idempotent replay;
* same identity with a different value **rejects the whole batch** (nothing
  written, including the batch's other rows);
* a channel is retracted by a higher-revision tombstone, and restored by a yet
  higher revision with a value.

Genuinely disagreeing channels are *not* an import error: each original
reading is retained so the conflict can be traced, and the last arrival never
silently wins.

### Per-record, per-target conclusions

`solve` consumes the current effective readings and the **published** plan
contracts and judges each requested output independently:

| status | meaning | payload |
| --- | --- | --- |
| `decided` | one exact value under every solution of the readings | `value` plus a `witness`: exact multipliers over the original readings that reproduce the target's coefficients and value |
| `nonunique` | at least two solutions satisfy every reading but disagree on the target | two concrete `solutions`, each with full assignments and per-reading residuals |
| `conflict` | the readings are inconsistent | a set of reading equations whose column coefficients cancel to `0` but whose right-hand sides sum to a non-zero residual |

A single undetermined column never hides a different identifiable target, and a
column that appears in no reading is never guessed from a default.  A direct
column reading is reported as a sourced equation (`direct_column`) like any
other.

For `a + b = 7`, target `a + b + 2` is decided `9` while target `a` is
non-unique (`(a,b) = (0,7)` and `(1,6)` are two examples).  Adding
`a - b + 1 = 2` decides `a = 4`, `b = 3`; a third channel reporting
`2a + 2b = 15` conflicts; withdrawing it restores identifiability.

### Proofs are immutable and self-verifying

Every `solve` persists a proof bound to the reading snapshot, the scheme digest
and the target plan digest.  `joint.reopen(proof_id)` re-reads it after a fresh
connection and re-checks it from its own contents (`verification["ok"]`): a
witness is re-expanded from the original readings, both solutions are
substituted into every reading, and a conflict combination is re-summed.
Rejected/later conclusions never overwrite an earlier proof.

## Scenario (materials/746)

`python3.11 run_scenario.py` executes the input's operations and the prose
follow-ups:

| Step | z (g) | w (g) |
| --- | --- | --- |
| initial (`p1`, r1 rev1) | 4 | 16 |
| r1.b → 10 (rev2) | 4 (b is zero-dep) | 28 |
| new product `p2` (`z = u − 2b`) | −8 | 28 |
| r1 rev3 `{a:3}` — request `{z}` | 4 | — |
| r1 rev3 — request `{z,w}` | rejected: `r1.b → w` | |
| r1 rev3 — `p2`, request `{z}` | rejected: `r1.b → z` | |
| r1 rev4 restores `b=10` — `p2 {z}` | −8 | |

The original `p1` results stay at 4 throughout.

`python3.11 run_joint_scenario.py` demonstrates the joint-reading flow:
non-unique (`a + b = 7` only), redundancy, a one-line contradiction from an
independent channel, and re-verification after withdrawal on a reopened
connection.

## Tests

```bash
python3.11 -m unittest discover -s tests -v
```

78 tests cover exact rationals, DAG validation, zero-dependency elimination,
shared-node algebra, cancellation explanation on a wide DAG, fact replay rules,
per-output-set eligibility, immutable result identity, incremental recompute
with group migration, persistence across connections, a 50-node / 10-output /
5000-fact run, and the joint-reading layer: immutable schemes, late/retract/
equivalent-fraction imports, all-or-nothing batch rejection, per-target
identifiability with witnesses, two-solution non-uniqueness, conflict
equations, direct-column equations, and proof reopen/verification.

## Scale

Documented bounds: ≤ 8 inputs, ≤ 50 nodes, ≤ 10 outputs, ≤ 5000 records per
plan; ≤ 8 columns and ≤ 32 channels per measurement scheme (hence ≤ 32 readings
per local recovery problem); coefficient and value numerator/denominator
magnitudes to 1 000 000; no output truncation.  Out of scope by design:
nonlinear functions, cycle solving, decimal expression type inference, optimal
incremental scheduling.
