// Test that we check supertrait bounds for WFedness.

#![feature(associated_type_defaults)]

#![allow(dead_code)]

trait ExtraCopy<T:Copy> { }

trait SomeTrait<T,U> //~ ERROR E0277
    where T: ExtraCopy<U>
{
}


fn main() { }
