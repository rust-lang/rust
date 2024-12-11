//@ proc-macro: test-macros.rs
//@ compile-flags: -Z span-debug
//@ check-pass

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use] extern crate test_macros;

include!("pretty-print-hack/rental-0.5.6/src/lib.rs");

fn main() {}
