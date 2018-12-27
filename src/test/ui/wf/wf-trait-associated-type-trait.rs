// Test that we check associated type default values for WFedness.

#![feature(associated_type_defaults)]

#![allow(dead_code)]

struct IsCopy<T:Copy> { x: T }

trait SomeTrait {
    type Type1;
    type Type2 = IsCopy<Self::Type1>;
    //~^ ERROR E0277
}


fn main() { }
