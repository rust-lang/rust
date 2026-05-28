//@ compile-flags:--test
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

/// ```
/// # #![cfg_attr(not(dox), deny(missing_abi,
/// # non_ascii_idents))]
///
/// pub struct Bar;
/// ```
pub struct Bar;
