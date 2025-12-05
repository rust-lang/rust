// This test ensures that the doctest will not use `#[allow(unused)]`.

//@ compile-flags:--test
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ failure-status: 101

#![doc(test(attr(allow(unused_variables), deny(warnings))))]

/// Example
///
/// ```rust,no_run
/// trait T { fn f(); }
/// ```
pub fn f() {}
