//@ compile-flags: -Znext-solver
#![feature(const_trait_impl)]

pub const trait Tr {
    fn a(&self) {}

    fn b(&self) {
        ().a()
        //~^ ERROR the trait bound `(): [const] Tr` is not satisfied
    }
}

impl Tr for () {}

fn main() {}
