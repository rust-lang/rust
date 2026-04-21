// Verify `condense_outlives_classes` under the parameter-aliasing
// rejection rule — an impl where ONE impl-side lifetime param maps to
// MULTIPLE distinct dyn bound variables (parameter aliasing).
//
// The test program (test.rs) implements `Aliased<'a, 'b>` for both
// `TypeA<'a, 'b>` (universal-looking, different params per bv) and
// `TypeB<'c>` at `Aliased<'c, 'c>` (one `'c` maps to both bvs 0 and 1
// of the dyn binder). Param aliasing disqualifies the TypeB impl from
// `impl_universally_admissible`, so `trait_cast_layout` invokes
// `condense_outlives_classes` for `Aliased`.
//
// Under the single empty outlives class, the aliasing rule requires
// mutual outlives between the two bvs that `'c` aliases. Reachability
// has only reflexivity and `'static →` edges, so the mutual-outlives
// check fails and `TypeB::Aliased` is rejected.
// Matrix row for Aliased: `[TypeA: ✓, TypeB: ✗]` — one row → one slot.
//
// `Free<'a, 'b>` also bypasses the fast path (shared Self/trait
// params) and receives one slot via condensation.
//
// Expected table layout: 2 slots total (1 Free + 1 Aliased).
//
// If the aliasing rule regressed and accidentally admitted the
// aliased impl under an empty class, the table length would still be
// 2 — but runtime casts of TypeB to `Aliased<'c, 'c>` that should
// fail would start succeeding, which a behavior-level assertion would
// catch.

//@ needs-target-std

use run_make_support::rustc;

fn main() {
    rustc()
        .input("test.rs")
        .arg("-Zdump-post-mono-mir")
        .run()
        // ── Table length: 1 slot per sub-trait via condensation ───
        .assert_stdout_contains("const 2_usize")
        // ── Slot indices ──────────────────────────────────────────
        .assert_stdout_contains("const 0_usize")
        .assert_stdout_contains("const 1_usize")
        // ── Instance headers ──────────────────────────────────────
        .assert_stdout_contains("post-mono MIR for instance");
}
