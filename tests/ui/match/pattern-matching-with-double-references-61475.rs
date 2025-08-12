// https://github.com/rust-lang/rust/issues/61475
//@ run-pass
#![allow(dead_code)]

enum E {
    A, B
}

fn main() {
    match &&E::A {
        &&E::A => {
        }
        &&E::B => {
        }
    };
}
