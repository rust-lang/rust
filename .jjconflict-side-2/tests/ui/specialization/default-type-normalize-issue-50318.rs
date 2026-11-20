//@ check-fail
//@ known-bug: #50318

#![feature(specialization)]
#![allow(incomplete_features)]

// Tests that we can normalize a `default type`.

trait Trait {
    type AssocType;
}

struct Struct {}

impl Trait for Struct {
    default type AssocType = i32;
}

type AssocType = <Struct as Trait>::AssocType;

fn main() {
    assert_eq!(std::any::type_name::<AssocType>(), "i32");
    let x: AssocType = 0;
}
