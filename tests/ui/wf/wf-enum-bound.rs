// Test that we check enum bounds for WFedness.

#![feature(associated_type_defaults)]

#![allow(dead_code)]

trait ExtraCopy<T:Copy> { }

enum SomeEnum<T,U>
    where T: ExtraCopy<U> //~ ERROR E0277
{
    SomeVariant(T,U)
}


fn main() { }
