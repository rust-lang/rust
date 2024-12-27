//@ check-pass
//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

#[const_trait]
pub trait Owo<X = <Self as Uwu>::T> {}

#[const_trait]
pub trait Uwu: Owo {
    type T;
}

fn main() {}
