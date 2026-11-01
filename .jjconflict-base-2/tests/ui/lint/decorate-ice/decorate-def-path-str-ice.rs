// Checks that the following does not ICE.
//
// Previously, this test ICEs when the `unused_must_use` lint is suppressed via the combination of
// `-A warnings` and `--cap-lints=warn`, because:
//
// - Its lint diagnostic struct `UnusedDef` implements `Diagnostic` manually and in the impl
//   `def_path_str` was called (which calls `trimmed_def_path`, which will produce a
//   `must_produce_diag` ICE if a trimmed def path is constructed but never emitted in a diagnostic
//   because it is expensive to compute).
// - A `Diagnostic` has an `into_diag` method which generates a `Diag` with (potentially)
//   lint-specific information.
// - The `into_diag` method is called as late as possible until `diag_lint_level` is called.
// - If a lint's corresponding diagnostic is suppressed (to be effectively allow at the final
//   emission time) via `-A warnings` or `--cap-lints=allow` (or `-A warnings` + `--cap-lints=warn`
//   like in this test case), `into_diag` is still called and a diagnostic is still constructed --
//   but the diagnostic is never eventually emitted, triggering the aforementioned
//   `must_produce_diag` ICE due to use of `trimmed_def_path`.
//
// Issue: <https://github.com/rust-lang/rust/issues/121774>.

//@ compile-flags: -Dunused_must_use -Awarnings --cap-lints=warn --crate-type=lib
//@ check-pass

#[must_use]
fn f() {}

pub fn g() {
    f();
}
