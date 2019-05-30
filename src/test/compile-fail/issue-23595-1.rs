#![feature(associated_type_defaults)]

use std::ops::{Index};

trait Hierarchy {
    type Value;
    type ChildKey;
    type Children = dyn Index<Self::ChildKey, Output=dyn Hierarchy>;
    //~^ ERROR: the value of the associated types `Value` (from the trait `Hierarchy`), `ChildKey`

    fn data(&self) -> Option<(Self::Value, Self::Children)>;
}

fn main() {}
