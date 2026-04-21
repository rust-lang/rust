// Verify that the trait cast table layout computed for a basic downcast
// scenario (3 sub-traits, 2 concrete types, no lifetimes) produces the
// expected resolved constants in post-monomorphization MIR.
//
// Expected table layout for `dyn Base`:
//   - 3 slots (one per sub-trait: Greet, Count, Describe)
//   - All impls universally admissible (no lifetimes → fast path)
//   - TypeA implements all 3 → all slots populated
//   - TypeB implements Greet + Count only → Describe slot is None
//
// The test checks resolved intrinsic constants in the dumped MIR:
//   - trait_metadata_table_len  → 3_usize
//   - trait_metadata_index      → slot indices 0, 1, 2
//   - is_lifetime_erasure_safe  → true (no lifetimes)

//@ needs-target-std

use run_make_support::rustc;

fn main() {
    // Compile with -Z dump-post-mono-mir (no path = stdout).
    // This dumps all post-monomorphization MIR bodies, including the
    // patched bodies where trait cast intrinsics are resolved to constants.
    rustc()
        .input("test.rs")
        .arg("-Zdump-post-mono-mir")
        .run()
        // ── Table length ──────────────────────────────────────────
        // trait_metadata_table_len<dyn Base>() resolves to 3_usize
        // (one slot per sub-trait: Greet, Count, Describe).
        .assert_stdout_contains("const 3_usize")
        // ── Erasure safety ────────────────────────────────────────
        // All three sub-traits have no lifetime binder variables, so
        // trait_cast_is_lifetime_erasure_safe resolves to `true`.
        .assert_stdout_contains("const true")
        // ── Slot indices ──────────────────────────────────────────
        // Three distinct slot indices (0, 1, 2) are assigned, one per
        // sub-trait. Each appears as a usize constant in a resolved
        // trait_metadata_index tuple: (_X = (<crate_id>, N_usize)).
        .assert_stdout_contains("const 0_usize")
        .assert_stdout_contains("const 1_usize")
        .assert_stdout_contains("const 2_usize")
        // ── Instance headers ──────────────────────────────────────
        // The dump should contain post-mono MIR headers.
        .assert_stdout_contains("post-mono MIR for instance");
}
