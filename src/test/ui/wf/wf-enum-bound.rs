// Test that we check enum bounds for WFedness.

#![feature(associated_type_defaults)]
#![feature(rustc_attrs)]
#![allow(dead_code)]

trait ExtraCopy<T:Copy> { }

enum SomeEnum<T,U> //~ ERROR E0277
    where T: ExtraCopy<U>
{
    SomeVariant(T,U)
}

#[rustc_error]
fn main() { }
