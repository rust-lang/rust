#![allow(bare_trait_objects)]
trait Item {
    type Core;
}
pub struct Flatten<I> {
    inner: <IntoIterator<Item: IntoIterator<Item: >>::IntoIterator as Item>::Core,
    //~^ ERROR E0191
}

fn main() {}
