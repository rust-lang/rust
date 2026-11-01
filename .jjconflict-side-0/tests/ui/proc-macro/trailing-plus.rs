//@ check-pass
//@ proc-macro: test-macros.rs
//@ compile-flags: -Z span-debug

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

extern crate test_macros;

#[test_macros::print_attr]
fn foo<T>() where T: Copy + {
}

fn main() {}
