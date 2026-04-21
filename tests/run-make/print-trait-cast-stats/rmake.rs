// Verify the `-Z print-trait-cast-stats` diagnostic flag.
//
// The flag (defined in compiler/rustc_session/src/options.rs) emits a
// single compact summary block describing the trait-cast monomorphization
// pipeline to stderr. It is implemented in
// compiler/rustc_monomorphize/src/partitioning.rs (`print_trait_cast_stats`).
//
// This test verifies:
//   1. `-Zprint-trait-cast-stats` emits the header block with the expected
//      labels.
//   2. Without the flag, none of those labels appear (flag-off fast path).

//@ needs-target-std

use run_make_support::rustc;

fn main() {
    // ---- 1. flag on --------------------------------------------------------
    rustc()
        .input("test.rs")
        .arg("-Zprint-trait-cast-stats")
        .run()
        // Header.
        .assert_stderr_contains("trait-cast stats:")
        // Counter labels (we don't assert specific numbers — those are not
        // load-bearing and would churn with unrelated pipeline changes).
        .assert_stderr_contains("delayed codegen entries:")
        .assert_stderr_contains("root supertraits:");

    // ---- 2. flag off (negative case) ---------------------------------------
    // With no `-Z print-trait-cast-stats`, the header must not appear.
    rustc().input("test.rs").run().assert_stderr_not_contains("trait-cast stats:");
}
