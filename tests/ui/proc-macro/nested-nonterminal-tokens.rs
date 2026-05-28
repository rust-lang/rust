//@ check-pass
//@ edition:2018
//@ compile-flags: -Z span-debug
//@ proc-macro: test-macros.rs

// Tests that we properly pass tokens to proc-macro when nested
// nonterminals are involved.

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use]
extern crate test_macros;


macro_rules! wrap {
    (first, $e:expr) => { wrap!(second, $e + 1) };
    (second, $e:expr) => { wrap!(third, $e + 2) };
    (third, $e:expr) => {
        print_bang!($e + 3)
    };
}

fn main() {
    let _ = wrap!(first, 0);
}
