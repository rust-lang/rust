// Same test as dead-code-module but with 2 doc(test(attr())) at different levels.

//@ edition: 2024
//@ compile-flags:--test
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout: "ran in \d+\.\d+s" -> "ran in $$TIME"
//@ normalize-stdout: "compilation took \d+\.\d+s" -> "compilation took $$TIME"
//@ failure-status: 101

#![doc(test(attr(allow(unused_variables))))]

mod my_mod {
    #![doc(test(attr(deny(warnings))))]

    /// Example
    ///
    /// ```rust,no_run
    /// trait T { fn f(); }
    /// ```
    pub fn f() {}
}

/// Example
///
/// ```rust,no_run
/// trait OnlyWarning { fn no_deny_warnings(); }
/// ```
pub fn g() {}
