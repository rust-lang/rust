// Ensure that `cfg_attr_trace` attributes aren't observable by proc-macros.

//@ check-pass
//@ proc-macro: test-macros.rs

#![feature(cfg_eval)]

#[macro_use]
extern crate test_macros;

#[cfg_eval]
#[test_macros::print_attr]
#[cfg_attr(false, test_macros::print_attr)]
#[cfg_attr(true, test_macros::print_attr)]
struct S;

#[cfg_eval]
#[test_macros::print_attr]
#[cfg(true)]
struct Z;

fn main() {}
