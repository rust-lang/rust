// Verify the `-Z dump-trait-cast-sensitivity` diagnostic flag.
//
// The flag (defined in compiler/rustc_session/src/options.rs) dumps
// per-instance trait-cast sensitivity metadata to stderr. It is implemented
// in compiler/rustc_monomorphize/src/cast_sensitivity.rs
// (`dump_trait_cast_sensitivity`), invoked at the end of
// `compute_cast_relevant_lifetimes`.
//
// This test verifies:
//   1. `-Zdump-trait-cast-sensitivity=all` emits the expected header and
//      mappings section for at least one instance.
//   2. A substring filter matching `exercise` still produces the dump for
//      that instance (filter path exercised, positive case).
//   3. A substring filter matching no instance produces no dump (negative
//      case — no `=== Sensitivity:` header).

//@ needs-target-std

use run_make_support::rustc;

fn main() {
    // ---- 1. filter = "all" --------------------------------------------------
    rustc()
        .input("test.rs")
        .arg("-Zdump-trait-cast-sensitivity=all")
        .run()
        // Section header for at least one sensitive instance.
        .assert_stderr_contains("=== Sensitivity:")
        // Mappings section (non-empty because at least one instance is
        // transitively sensitive in this test).
        .assert_stderr_contains("Mappings (");

    // ---- 2. filter = substring of an instance's printed name ---------------
    // `exercise` is both directly and transitively sensitive in `test.rs`.
    rustc()
        .input("test.rs")
        .arg("-Zdump-trait-cast-sensitivity=exercise")
        .run()
        .assert_stderr_contains("=== Sensitivity:")
        .assert_stderr_contains("exercise");

    // ---- 3. filter matching no instance (negative case) --------------------
    rustc()
        .input("test.rs")
        .arg("-Zdump-trait-cast-sensitivity=NoSuchFunctionZZZ")
        .run()
        .assert_stderr_not_contains("=== Sensitivity:");
}
