// Test that we check where-clauses on inherent impl methods.

#![feature(associated_type_defaults)]

#![allow(dead_code)]

trait ExtraCopy<T:Copy> { }

struct Foo<T,U>(T,U);

impl<T,U> Foo<T,U> {
    fn foo(self) where T: ExtraCopy<U> //~ ERROR E0277
    {}
}


fn main() { }
