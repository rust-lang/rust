//@ check-pass
//@ compile-flags:--test -Zunstable-options --nocapture
//@ normalize-stderr: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"

/// ```compile_fail
/// fn foo() {
///     Input: 123
/// }
/// ```
pub struct Foo;
