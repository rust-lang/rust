// This test ensures that the 2024 edition merged doctest will not use `#[allow(unused)]`.

//@ edition: 2024
//@ compile-flags:--test
//@ normalize-stdout-test: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout-test: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ failure-status: 101

#![doc(test(attr(allow(unused_variables), deny(warnings))))]

/// Example
///
/// ```rust,no_run
/// trait T { fn f(); }
/// ```
pub fn f() {}
