//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

#[const_trait]
pub trait Tr {
    (const) fn a(&self) {}

    (const) fn b(&self) {
        ().a()
        //~^ ERROR the trait bound `(): ~const Tr` is not satisfied
    }
}

impl Tr for () {}

fn main() {}
