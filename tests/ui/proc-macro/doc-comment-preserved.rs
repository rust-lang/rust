//@ check-pass
//@ compile-flags: -Z span-debug
//@ proc-macro: test-macros.rs

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use]
extern crate test_macros;

print_bang! {

/**
*******
* DOC *
* DOC *
* DOC *
*******
*/
pub struct S;

}

fn main() {}
