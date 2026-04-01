//@ check-pass
//@ proc-macro: test-macros.rs
//@ compile-flags: -Z span-debug

#![feature(rustc_attrs)]

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use]
extern crate test_macros;

// Tests the pretty-printing behavior of various (unparsed) tokens
print_bang_consume!({
    #![rustc_dummy]
    let a = "hello".len();
    matches!(a, 5);
});

fn main() {}
