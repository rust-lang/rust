// edition:2018
// check-pass
#![allow(unused)]
#![deny(future_prelude_collision)]

struct S;

impl S {
    fn try_into(self) -> S { S }
}

// See https://github.com/rust-lang/rust/issues/86633
fn main() {
    let s = S;
    let s2 = s.try_into();
}
