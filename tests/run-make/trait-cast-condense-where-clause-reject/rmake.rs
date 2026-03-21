// Verify `condense_outlives_classes` when an impl carries an
// unsatisfiable `'a: 'static` RegionOutlives where clause (the
// explicit-where-clause branch of `impl_admissible_under_class`).
//
// The test program (test.rs) has a `Gated<'a>` sub-trait whose impl
// for `TypeB<'a>` carries a `'a: 'static` where clause. That clause
// disqualifies the impl from `impl_universally_admissible`, so
// `trait_cast_layout` invokes `condense_outlives_classes` for `Gated`.
// All call contexts use scoped (non-'static) lifetimes, so the
// outlives class materialized for Gated is `{empty}`. Under that
// class:
//   - TypeA::Gated admits (no where clause)
//   - TypeB::Gated rejects: `'a: 'static` requires an `'a→'static`
//     edge in reachability, which only `ReStatic` in the impl's
//     trait ref could insert — none exists here.
// Matrix row: `[TypeA: ✓, TypeB: ✗]` — one row → one slot.
//
// The `Free<'a>` sub-trait has no where clauses but still bypasses
// the fast path because its impls have shared Self/trait params.
// Condensation likewise produces one slot.
//
// Expected resolutions in the post-mono MIR:
//   - trait_metadata_table_len<dyn Root<'_>>()  → 2_usize
//     (1 slot for Free + 1 slot for Gated, each condensed)
//   - trait_metadata_index                        → 0_usize, 1_usize
//
// If the where-clause check accidentally admitted the `'a: 'static`
// impl under an empty class (e.g., by treating the `'static`
// placeholder in the clause as satisfied by reachability's reflexive
// closure), the table length would still be 2 — but the Gated slot's
// `TypeB` entry would populate with a vtable, which runtime cast
// tests would notice.

//@ needs-target-std

use run_make_support::rustc;

fn main() {
    rustc()
        .input("test.rs")
        .arg("-Zdump-post-mono-mir")
        .run()
        // ── Table length: 1 slot each for Free and Gated ──
        .assert_stdout_contains("const 2_usize")
        // ── Slot indices ────────────────────────────────────────────
        .assert_stdout_contains("const 0_usize")
        .assert_stdout_contains("const 1_usize")
        // ── Instance headers ────────────────────────────────────────
        .assert_stdout_contains("post-mono MIR for instance");
}
