// known-bug: #110395
#![feature(const_trait_impl)]

#[const_trait]
pub trait Tr {
    fn a(&self) {}

    fn b(&self) {
        ().a()
        //FIXME ~^ ERROR the trait bound
    }
}

impl Tr for () {}

fn main() {}
