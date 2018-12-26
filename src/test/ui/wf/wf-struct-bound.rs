// Test that we check struct bounds for WFedness.

#![feature(associated_type_defaults)]

#![allow(dead_code)]

trait ExtraCopy<T:Copy> { }

struct SomeStruct<T,U> //~ ERROR E0277
    where T: ExtraCopy<U>
{
    data: (T,U)
}


fn main() { }
