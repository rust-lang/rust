// Test that we check associated type default values for WFedness.

#![feature(associated_type_defaults)]
#![feature(rustc_attrs)]
#![allow(dead_code)]

trait SomeTrait<'a> {
    type Type1;
    type Type2 = &'a Self::Type1;
    //~^ ERROR E0309
}

#[rustc_error]
fn main() { }
