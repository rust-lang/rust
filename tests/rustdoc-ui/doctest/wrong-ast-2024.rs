//@ edition: 2024
//@ compile-flags:--test --test-args=--test-threads=1
//@ normalize-stdout-test: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout-test: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout-test: ".rs:\d+:\d+" -> ".rs:$$LINE:$$COL"
//@ failure-status: 101

/// ```
/// /* plop
/// ```
pub fn one() {}

/// ```
/// } mod __doctest_1 { fn main() {
/// ```
pub fn two() {}

/// ```should_panic
/// panic!()
/// ```
pub fn three() {}
