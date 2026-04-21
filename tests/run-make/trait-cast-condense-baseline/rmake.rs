// Verify the condensation code path (NOT the fast path) when a
// lifetime-parameterized trait has impls whose Self type and trait
// ref share parameters — a shape that disables
// `impl_universally_admissible`.
//
// Every impl in the test program (test.rs) has the form
// `impl<'a, 'b> SubX<'a, 'b> for TypeA<'a, 'b>`; the lifetime params
// are shared between Self and trait args. `impl_universally_admissible`
// therefore returns false for every impl, so `trait_cast_layout`
// bypasses the fast path and invokes `condense_outlives_classes` for
// every sub-trait.
//
// With a single (empty) outlives class materialized per sub-trait,
// condensation produces exactly one group per sub-trait. The table
// length is therefore `num_sub_traits == 2`. This is the baseline
// "condensation actually runs and produces minimum slots" contract: if
// condensation regressed and started emitting multiple slots for a
// single class, the table length would grow and this test would fail.
//
// Expected resolutions in the post-mono MIR:
//   - trait_metadata_table_len<dyn Root<'_,'_>>()  → 2_usize
//   - trait_metadata_index for SubX and SubY       → 0_usize, 1_usize
//
// (Erasure safety is orthogonal to condensation and this test does not
// pin `trait_cast_is_lifetime_erasure_safe`.)

//@ needs-target-std

use run_make_support::rustc;

fn main() {
    rustc()
        .input("test.rs")
        .arg("-Zdump-post-mono-mir")
        .run()
        // ── Table length ──────────────────────────────────────────
        // One slot per sub-trait — condensation collapses the single
        // materialized class to a single group.
        .assert_stdout_contains("const 2_usize")
        // ── Slot indices ──────────────────────────────────────────
        // Two distinct slot indices (0, 1), one per sub-trait.
        .assert_stdout_contains("const 0_usize")
        .assert_stdout_contains("const 1_usize")
        // ── Instance headers ──────────────────────────────────────
        .assert_stdout_contains("post-mono MIR for instance");
}
