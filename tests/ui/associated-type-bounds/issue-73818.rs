// Test that associated type bounds are correctly normalized when checking
// default associated type values.
//@ check-pass

#![allow(incomplete_features)]
#![feature(specialization)]

#[derive(PartialEq)]
enum Never {}
trait Foo {
    type Assoc: PartialEq; // PartialEq<<Self as Foo>::Assoc>
}
impl<T> Foo for T {
    default type Assoc = Never;
}

trait Trait1 {
    type Selection: PartialEq;
}
trait Trait2: PartialEq<Self> {}
impl<T: Trait2> Trait1 for T {
    default type Selection = T;
}

fn main() {}
