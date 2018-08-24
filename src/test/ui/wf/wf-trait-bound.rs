// Test that we check supertrait bounds for WFedness.

#![feature(associated_type_defaults)]
#![feature(rustc_attrs)]
#![allow(dead_code)]

trait ExtraCopy<T:Copy> { }

trait SomeTrait<T,U> //~ ERROR E0277
    where T: ExtraCopy<U>
{
}

#[rustc_error]
fn main() { }
