//@ compile-flags: -Znext-solver
#![allow(incomplete_features)]
#![feature(const_trait_impl, effects)]

#[const_trait]
pub trait Tr {
    fn a(&self) {}

    fn b(&self) {
        ().a()
        //~^ ERROR the trait bound `(): ~const Tr` is not satisfied
    }
}

impl Tr for () {}

fn main() {}
