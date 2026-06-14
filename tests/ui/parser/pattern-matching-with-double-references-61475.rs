// https://github.com/rust-lang/rust/issues/61475
//@ check-pass
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
