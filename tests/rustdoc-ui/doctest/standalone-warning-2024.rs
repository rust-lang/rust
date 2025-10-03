// This test checks that it will output warnings for usage of `standalone` or `standalone_crate`.

//@ edition: 2024
//@ compile-flags:--test
//@ normalize-stdout: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout: ".rs:\d+:\d+" -> ".rs:$$LINE:$$COL"

#![deny(warnings)]

//! ```standalone
//~^ ERROR unknown attribute `standalone`
//~| ERROR unknown attribute `standalone-crate`
//! bla
//! ```
//!
//! ```standalone-crate
//! bla
//! ```
