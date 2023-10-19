// check-pass
#![feature(const_trait_impl, effects)]

pub trait Owo<X = <Self as Uwu>::T> {}

#[const_trait]
pub trait Uwu: Owo {
    type T;
}

fn main() {}
