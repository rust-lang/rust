//@ compile-flags:--test
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ check-pass

#![no_std]

extern crate alloc;

/// ```
/// assert!(true)
/// ```
pub fn f() {}
