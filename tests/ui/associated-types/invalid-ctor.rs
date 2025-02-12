//@ run-rustfix

#![allow(unused)]

struct Constructor(i32);

trait Trait {
    type Out;

    fn mk() -> Self::Out;
}

impl Trait for () {
    type Out = Constructor;

    fn mk() -> Self::Out {
        Self::Out(1)
        //~^ ERROR no associated item named `Out` found for unit type `()`
    }
}

fn main() {}
