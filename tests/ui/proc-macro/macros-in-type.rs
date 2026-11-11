//@ check-pass
//@ proc-macro: test-macros.rs

#[macro_use]
extern crate test_macros;

const C: identity!(u8) = 10;

fn main() {
    let c: u8 = C;
}
