// Test that we check associated type default values for WFedness.

#![feature(associated_type_defaults)]

#![allow(dead_code)]

trait SomeTrait<'a> {
    type Type1;
    type Type2 = &'a Self::Type1;
    //~^ ERROR E0309
}


fn main() { }
