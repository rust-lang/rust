// Verify the `-Z dump-trait-cast-chain-composition` diagnostic flag.
//
// The flag (defined in compiler/rustc_session/src/options.rs) dumps per-link
// details of trait-cast call_id chain composition to stderr. It is implemented
// in compiler/rustc_monomorphize/src/cast_sensitivity.rs inside
// `compose_all_through_chain`.
//
// This test verifies:
//   1. `-Zdump-trait-cast-chain-composition=all` emits the expected header,
//      at least one `Link ` entry, and the `Final mapping` section.
//   2. A substring filter matching `exercise` still produces the dump when the
//      caller instance name contains it (filter path exercised, positive case).
//   3. A substring filter matching no caller produces no dump (negative case —
//      no `=== Chain Composition:` header).

//@ needs-target-std

use run_make_support::rustc;

fn main() {
    // ---- 1. filter = "all" --------------------------------------------------
    rustc()
        .input("test.rs")
        .arg("-Zdump-trait-cast-chain-composition=all")
        .run()
        // Header for at least one invocation.
        .assert_stderr_contains("=== Chain Composition:")
        // Per-link details — at least one link is emitted.
        .assert_stderr_contains("Link ")
        // Final mapping section is present.
        .assert_stderr_contains("Final mapping");

    // ---- 2. filter = substring of a caller's printed name ------------------
    // `exercise` is the caller on at least one chain composition invocation
    // (it calls `core::cast!`, which expands to sensitive intrinsics).
    rustc()
        .input("test.rs")
        .arg("-Zdump-trait-cast-chain-composition=exercise")
        .run()
        .assert_stderr_contains("=== Chain Composition:")
        .assert_stderr_contains("exercise");

    // ---- 3. filter matching no caller (negative case) ----------------------
    rustc()
        .input("test.rs")
        .arg("-Zdump-trait-cast-chain-composition=ZZZNoMatchZZZ")
        .run()
        .assert_stderr_not_contains("=== Chain Composition:");
}
