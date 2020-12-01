// Regression test for issue #75734
// Ensures that we don't lose tokens when pretty-printing would
// normally insert extra parentheses.

// check-pass
// aux-build:test-macros.rs
// compile-flags: -Z span-debug

#![no_std] // Don't load unnecessary hygiene information from std
extern crate std;

#[macro_use]
extern crate test_macros;

macro_rules! mul_2 {
    ($val:expr) => {
        print_bang!($val * 2);
    };
}


#[print_attr]
fn main() {
    &|_: u8| {};
    mul_2!(1 + 1);
}
