//! Regression test for https://github.com/rust-lang/rust/issues/2550

//@ run-pass
#![allow(dead_code)]
#![allow(non_snake_case)]

struct C {
    x: usize,
}

fn C(x: usize) -> C {
    C { x }
}

fn f<T>(_x: T) {}

pub fn main() {
    f(C(1));
}
