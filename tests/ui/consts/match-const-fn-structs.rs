// run-pass
#![allow(unused_variables)]

// https://github.com/rust-lang/rust/issues/46114

#[derive(Eq, PartialEq)]
struct A { value: u32 }

const fn new(value: u32) -> A {
    A { value }
}

const A_1: A = new(1);
const A_2: A = new(2);

fn main() {
    let a_str = match new(42) {
        A_1 => "A 1",
        A_2 => "A 2",
        _ => "Unknown A",
    };
}
