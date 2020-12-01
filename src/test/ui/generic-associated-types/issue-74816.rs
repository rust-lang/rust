#![feature(associated_type_defaults)]
#![feature(generic_associated_types)]
#![allow(incomplete_features)]

trait Trait1 {
    fn foo();
}

trait Trait2 {
    type Associated: Trait1 = Self;
    //~^ ERROR: the trait bound `Self: Trait1` is not satisfied
    //~| the size for values of type `Self` cannot be known
}

impl Trait2 for () {}

fn call_foo<T: Trait2>() {
    T::Associated::foo()
}

fn main() {
    call_foo::<()>()
}
