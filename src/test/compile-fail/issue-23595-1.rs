#![feature(associated_type_defaults)]

use std::ops::{Index};

trait Hierarchy {
    type Value;
    type ChildKey;
    type Children = Index<Self::ChildKey, Output=Hierarchy>;
    //~^ ERROR: the value of the associated type `ChildKey`
    //~^^ ERROR: the value of the associated type `Children`
    //~^^^ ERROR: the value of the associated type `Value`

    fn data(&self) -> Option<(Self::Value, Self::Children)>;
}

fn main() {}
