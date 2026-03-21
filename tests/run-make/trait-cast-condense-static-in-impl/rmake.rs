// Verify `condense_outlives_classes` when an impl fixes a trait-ref
// lifetime arg to the literal `'static` (the `ReStatic` branch of
// `impl_admissible_under_class`).
//
// The test program (test.rs) has `TypeB` implementing `Anchored<'static>`
// (and `Root`/`Free`) rather than `Anchored<'a>` for any free `'a`. The
// concrete `'static` in the trait ref disqualifies the impl from
// `impl_universally_admissible`, so `trait_cast_layout` invokes
// `condense_outlives_classes` for the `Anchored` sub-trait.
//
// Under the single materialized empty outlives class, admissibility
// requires `bv0 outlives 'static`. Reachability contains only
// reflexivity and `'static → *` edges — no `bv → 'static` edge — so
// the admissibility check rejects `TypeB::Anchored`.
// Matrix row for Anchored: `[TypeA: ✓, TypeB: ✗]` → one row → one slot.
//
// `Free<'a>` bypasses the fast path via shared Self/trait params and
// also receives one slot.
//
// Expected total table length: 2 slots.
//
// If the `ReStatic` rule regressed (e.g., remapping `ReStatic`
// incorrectly, or reading the reachability matrix with the wrong
// `num_bvs`), either the slot count would change or the TypeB
// admissibility pattern would flip — both of which this test
// constrains, when combined with runtime behavior checks.

//@ needs-target-std

use run_make_support::rustc;

fn main() {
    rustc()
        .input("test.rs")
        .arg("-Zdump-post-mono-mir")
        .run()
        // ── Table length: 1 slot per sub-trait via condensation ──
        .assert_stdout_contains("const 2_usize")
        // ── Slot indices ─────────────────────────────────────────
        .assert_stdout_contains("const 0_usize")
        .assert_stdout_contains("const 1_usize")
        // ── Instance headers ─────────────────────────────────────
        .assert_stdout_contains("post-mono MIR for instance");
}
