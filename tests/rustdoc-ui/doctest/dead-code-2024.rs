// This test ensures that the 2024 edition merged doctest will not use `#[allow(unused)]`.

//@ edition: 2024
//@ compile-flags:--test
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout: "ran in \d+\.\d+s" -> "ran in $$TIME"
//@ normalize-stdout: "compilation took \d+\.\d+s" -> "compilation took $$TIME"
//@ failure-status: 101

#![doc(test(attr(allow(unused_variables), deny(warnings))))]

/// Example
///
/// ```rust,no_run
/// trait T { fn f(); }
/// ```
pub fn f() {}
