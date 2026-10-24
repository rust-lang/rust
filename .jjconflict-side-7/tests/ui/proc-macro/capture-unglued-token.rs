//@ proc-macro: test-macros.rs
//@ compile-flags: -Z span-debug
//@ check-pass

// Tests that we properly handle parsing a nonterminal
// where we have two consecutive angle brackets (one inside
// the nonterminal, and one outside)

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;
extern crate test_macros;

macro_rules! trailing_angle {
    (Option<$field:ty>) => {
        test_macros::print_bang_consume!($field);
    }
}

trailing_angle!(Option<Vec<u8>>);
fn main() {}
