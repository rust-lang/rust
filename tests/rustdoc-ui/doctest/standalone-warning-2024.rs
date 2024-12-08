// This test checks that it will output warnings for usage of `standalone` or `standalone_crate`.

//@ edition: 2024
//@ compile-flags:--test
//@ normalize-stdout-test: "tests/rustdoc-ui/doctest" -> "$$DIR"
//@ normalize-stdout-test: "finished in \d+\.\d+s" -> "finished in $$TIME"
//@ normalize-stdout-test: ".rs:\d+:\d+" -> ".rs:$$LINE:$$COL"

#![deny(warnings)]

//! ```standalone
//! bla
//! ```
//!
//! ```standalone-crate
//! bla
//! ```
