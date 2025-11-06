#![allow(supertrait_item_shadowing_definition)]

//@ check-pass
pub trait Subscriber {
    type Input;
}

pub trait Processor: Subscriber<Input = <Self as Processor>::Input> {
    type Input;
}

fn main() {}
