#![allow(bare_trait_objects)]
#![feature(associated_type_bounds)]
trait Item {
    type Core;
}
pub struct Flatten<I> {
    inner: <IntoIterator<Item: IntoIterator<Item: >>::IntoIterator as Item>::Core,
    //~^ ERROR E0191
    //~| ERROR E0223
}

fn main() {}
