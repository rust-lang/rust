//@ check-pass

#![feature(trait_alias)]

mod alpha {
    pub trait A {}
    pub trait C = A;
}

#[allow(unused_imports)]
use alpha::C;

fn main() {}
