#![feature(marker_trait_attr)]
#[marker]
pub trait Marker {}

pub struct B;

trait NotMarker {
    type Assoc;
}

impl NotMarker for B {
    type Assoc = usize;
}

impl<T: Marker> NotMarker for T {
    //~^ ERROR conflicting implementations of trait `NotMarker` for type `B`
    type Assoc = Box<String>;
}

fn main() {}
