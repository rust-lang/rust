// run-pass

// Regression test for #47153: constants in a generic context (such as
// a trait) used to ICE.

#![allow(warnings)]

trait Foo {
    const B: bool = true;
}

struct Bar<T> { x: T }

impl<T> Bar<T> {
    const B: bool = true;
}

fn main() { }
