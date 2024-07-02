// This test ensures that if there is only one mergeable doctest, then it is
// instead run as part of standalone doctests.

//@ compile-flags:--test --test-args=--test-threads=1 -Zunstable-options --edition 2024
//@ normalize-stdout-test: "tests/rustdoc-ui" -> "$$DIR"
//@ normalize-stdout-test "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout-test ".rs:\d+:\d+" -> ".rs:$$LINE:$$COL"
//@ check-pass

/// ```
/// let x = 12;
/// ```
///
/// ```compile_fail
/// let y = x;
/// ```
pub fn one() {}
