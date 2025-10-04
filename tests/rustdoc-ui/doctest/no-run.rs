// This test ensures that the `--no-run` flag works the same between normal and merged doctests.
// Regression test for <https://github.com/rust-lang/rust/issues/143858>.

//@ check-pass
//@ revisions: edition2021 edition2024
//@ [edition2021]edition:2021
//@ [edition2024]edition:2024
//@ compile-flags:-Z unstable-options --test --no-run --test-args=--test-threads=1
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout: "ran in \d+\.\d+s" -> "ran in $$TIME"
//@ normalize-stdout: "compilation took \d+\.\d+s" -> "compilation took $$TIME"

/// ```
/// let a = true;
/// ```
/// ```should_panic
/// panic!()
/// ```
/// ```ignore (incomplete-code)
/// fn foo() {
/// ```
/// ```no_run
/// loop {
///     println!("Hello, world");
/// }
/// ```
/// fails to compile
/// ```compile_fail
/// let x = 5;
/// x += 2; // shouldn't compile!
/// ```
/// Ok the test does not run
/// ```
/// panic!()
/// ```
/// Ok the test does not run
/// ```should_panic
/// loop {
///     println!("Hello, world");
/// panic!()
/// }
/// ```
pub fn f() {}
