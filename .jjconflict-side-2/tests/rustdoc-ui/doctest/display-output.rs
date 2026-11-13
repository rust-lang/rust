// Test that `--show-output` has an effect and `allow(unused)` can be overridden.

//@ check-pass
//@ edition:2018
//@ compile-flags:--test --test-args=--show-output
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"

/// ```
/// #![warn(unused)]
/// let x = 12;
///
/// fn foo(x: &dyn std::fmt::Display) {}
/// ```
pub fn foo() {}
