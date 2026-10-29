//@ check-pass
//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

pub const trait Owo<X = <Self as Uwu>::T> {}

pub const trait Uwu: Owo {
    type T;
}

fn main() {}
