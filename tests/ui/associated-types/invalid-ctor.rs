// Related to #71054 / #120871: `Self::Out(1)` now resolves when Out is an associated
// type set to a tuple struct.
//
//@ check-pass
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
    }
}

fn main() {}
