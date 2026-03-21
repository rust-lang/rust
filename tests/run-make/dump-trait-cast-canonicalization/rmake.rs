// Verify the `-Z dump-trait-cast-canonicalization` diagnostic flag.
//
// The flag (defined in compiler/rustc_session/src/options.rs) dumps
// cascade canonicalization decisions to stderr. It is implemented in
// compiler/rustc_monomorphize/src/partitioning.rs (cascade_canonicalize):
// a depth-ordered walk that emits a header, per-depth Phase 1 (patch)
// and Phase 3 (emit) sections, Phase 2 (dedup) entries for
// signature groups of size > 1, and a final canon map summary.
//
// This test verifies:
//   1. `-Zdump-trait-cast-canonicalization` emits the expected header,
//      `Total delayed instances:` line, and per-depth emission
//      evidence (at least one `Depth 0:` or `Phase 1 (patch):` line).
//   2. Without the flag, none of the canonicalization-dump output is
//      emitted (negative case — no `=== Trait-Cast Canonicalization ===`).

//@ needs-target-std

use run_make_support::rustc;

fn main() {
    // ---- 1. flag on --------------------------------------------------------
    let out = rustc().input("test.rs").arg("-Zdump-trait-cast-canonicalization").run();
    out.assert_stderr_contains("=== Trait-Cast Canonicalization ===")
        .assert_stderr_contains("Total delayed instances:");
    // Per-depth emission proof: either the depth header or the Phase 1
    // subheader must appear. `Depth 0:` is always emitted whenever at
    // least one delayed Instance exists (the test program has several).
    let stderr = out.stderr_utf8();
    assert!(
        stderr.contains("Depth 0:") || stderr.contains("Phase 1 (patch):"),
        "expected per-depth emission marker in stderr, got:\n{stderr}"
    );

    // ---- 2. flag off (negative) --------------------------------------------
    rustc()
        .input("test.rs")
        .run()
        .assert_stderr_not_contains("=== Trait-Cast Canonicalization ===");
}
