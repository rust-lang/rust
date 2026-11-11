//@ known-bug: rust-lang/rust#125801

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

trait Foo {
    type Output;
}

impl Foo for [u8; 3] {
    type Output = [u8; 3];
}

static A: <[u8; N] as Foo>::Output = [1, 2, 3];

fn main() {
    || {
        let _ = A[1];
    };
}
