// Order of inert attributes, both built-in and custom is preserved during expansion.

//@ check-pass
//@ compile-flags: -Z span-debug
//@ proc-macro: test-macros.rs

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use]
extern crate test_macros;

/// 1
#[rustfmt::attr2]
#[doc = "3"]
#[print_attr(nodebug)]
#[doc = "4"]
#[rustfmt::attr5]
/// 6
#[print_attr(nodebug)]
struct S;

fn main() {}
