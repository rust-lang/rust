// Test that we check associated type bounds for WFedness.

#![feature(associated_type_defaults)]
#![feature(rustc_attrs)]
#![allow(dead_code)]

trait ExtraCopy<T:Copy> { }

trait SomeTrait<T> { //~ ERROR E0277
    type Type1: ExtraCopy<T>;
}

#[rustc_error]
fn main() { }
