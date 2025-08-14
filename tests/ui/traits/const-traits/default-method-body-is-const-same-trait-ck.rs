//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

#[const_trait]
pub trait Tr {
    fn a(&self) {}

    fn b(&self) {
        ().a()
        //~^ ERROR the trait bound `(): [const] Tr` is not satisfied
    }
}

impl Tr for () {}

fn main() {}
