//@ check-pass
//@ compile-flags:--test --test-args=--test-threads=1
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"

// Regression test for <https://github.com/rust-lang/rust/issues/131893>.
// It ensures that if a function called `main` is nested, it will not consider
// it as the `main` function.

/// ```
/// fn dox() {
///     fn main() {}
/// }
/// ```
pub fn foo() {}

// This one ensures that having a nested `main` doesn't prevent the
// actual `main` function to be detected.
/// ```
/// fn main() {
///     fn main() {}
/// }
/// ```
pub fn foo2() {}
