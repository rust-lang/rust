// edition:2018
// check-pass
#![allow(unused)]
#![deny(rust_2021_prelude_collisions)]

struct S;

impl S {
    fn try_into(self) -> S {
        S
    }
}

struct X;

trait Hey {
    fn from_iter(_: i32) -> Self;
}

impl Hey for X {
    fn from_iter(_: i32) -> Self {
        X
    }
}

fn main() {
    // See https://github.com/rust-lang/rust/issues/86633
    let s = S;
    let s2 = s.try_into();

    // See https://github.com/rust-lang/rust/issues/86902
    X::from_iter(1);
}
