// Verify the `-Z dump-trait-cast-augmentation` diagnostic flag.
//
// The flag (defined in compiler/rustc_session/src/options.rs) dumps
// per-edge augmentation decisions to stderr. It is implemented in
// compiler/rustc_monomorphize/src/cast_sensitivity.rs
// (`maybe_dump_augmentation`), invoked from every `augment_callee` call
// site once the final augmented Instance is known.
//
// This test verifies:
//   1. `-Zdump-trait-cast-augmentation=all` emits the expected section
//      header, caller-outlives-env line, and augmented-callee line for
//      at least one augmentation.
//   2. A substring filter matching `exercise` produces the dump whose
//      caller name contains that substring (filter path exercised,
//      positive case).
//   3. A substring filter matching no caller produces no dump (negative
//      case — no `=== Augmentation:` header).

//@ needs-target-std

use run_make_support::rustc;

fn main() {
    // ---- 1. filter = "all" --------------------------------------------------
    rustc()
        .input("test.rs")
        .arg("-Zdump-trait-cast-augmentation=all")
        .run()
        .assert_stderr_contains("=== Augmentation:")
        .assert_stderr_contains("Caller outlives env:")
        .assert_stderr_contains("Augmented callee:");

    // ---- 2. filter = substring of a caller's printed name ------------------
    // `exercise` is directly sensitive in `test.rs`, so it acts as a caller
    // whose augmentations are emitted for its sensitive callees.
    rustc()
        .input("test.rs")
        .arg("-Zdump-trait-cast-augmentation=exercise")
        .run()
        .assert_stderr_contains("=== Augmentation:")
        .assert_stderr_contains("exercise");

    // ---- 3. filter matching no caller (negative case) ----------------------
    rustc()
        .input("test.rs")
        .arg("-Zdump-trait-cast-augmentation=ZZZNoMatchZZZ")
        .run()
        .assert_stderr_not_contains("=== Augmentation:");
}
