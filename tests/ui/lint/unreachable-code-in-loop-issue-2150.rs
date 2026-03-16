// https://github.com/rust-lang/rust/issues/2150
// Tests that statements after panic!() are considered unreachable and raise a lint error.
#![deny(unreachable_code)]
#![allow(unused_variables)]
#![allow(dead_code)]

fn fail_len(v: Vec<isize> ) -> usize {
    let mut i = 3;
    panic!();
    for x in &v { i += 1; }
    //~^ ERROR: unreachable statement
    return i;
}
fn main() {}
