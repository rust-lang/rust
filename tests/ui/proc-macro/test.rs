//@ check-pass
//@ proc-macro: api/proc_macro_api_tests.rs
//@ edition: 2021
// Because of a (known) proc-macro ABI issue with GCC backend, this test
// fails because of it.
//@ ignore-backends: gcc

//! This is for everything that *would* be a #[test] inside of libproc_macro,
//! except for the fact that proc_macro objects are not capable of existing
//! inside of an ordinary Rust test execution, only inside a macro.

extern crate proc_macro_api_tests;

proc_macro_api_tests::run!();

fn main() {}
