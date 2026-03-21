// Verify the `-Z dump-trait-cast-erasure-safety` diagnostic flag.
//
// The flag (defined in compiler/rustc_session/src/options.rs) dumps
// per-query trait-cast erasure-safety analysis decisions to stderr. It is
// implemented in compiler/rustc_monomorphize/src/erasure_safe.rs
// (`dump_erasure_safety`), invoked at the end of the query provider
// `is_lifetime_erasure_safe`.
//
// This test verifies:
//   1. `-Zdump-trait-cast-erasure-safety=all` emits the expected header
//      and verdict for at least one query.
//   2. A substring filter matching the `GraphRoot` super-trait name
//      still produces the dump for that query (positive filter case).
//   3. A substring filter matching no super-trait name produces no dump
//      (negative case).

//@ needs-target-std

use run_make_support::rustc;

fn main() {
    // ---- 1. filter = "all" --------------------------------------------------
    rustc()
        .input("test.rs")
        .arg("-Zdump-trait-cast-erasure-safety=all")
        .run()
        // Section header for at least one erasure-safety query.
        .assert_stderr_contains("=== Erasure Safety:")
        // Verdict line always present — either `safe` or `unsafe (...)`.
        .assert_stderr_contains("Verdict:");

    // ---- 2. filter = substring of a super-trait printed name ---------------
    // `GraphRoot` is the super-trait for both casts in `test.rs`, so it
    // appears in at least one query's printed super-trait name.
    rustc()
        .input("test.rs")
        .arg("-Zdump-trait-cast-erasure-safety=GraphRoot")
        .run()
        .assert_stderr_contains("=== Erasure Safety:")
        .assert_stderr_contains("Verdict:")
        .assert_stderr_contains("GraphRoot");

    // ---- 3. filter matching no query (negative case) -----------------------
    rustc()
        .input("test.rs")
        .arg("-Zdump-trait-cast-erasure-safety=ZZZNoMatchZZZ")
        .run()
        .assert_stderr_not_contains("=== Erasure Safety:");
}
