#![feature(const_trait_impl, effects)] //~ WARN the feature `effects` is incomplete

#[const_trait]
pub trait Tr {
    fn a(&self) {}

    fn b(&self) {
        ().a()
        //~^ ERROR the trait bound
    }
}

impl Tr for () {}

fn main() {}
