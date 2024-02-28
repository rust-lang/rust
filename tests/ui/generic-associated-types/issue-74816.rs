#![feature(associated_type_defaults)]

trait Trait1 {
    fn foo();
}

trait Trait2 {
    type Associated: Trait1 = Self;
    //~^ ERROR trait `Trait1` is not implemented for `Self`
    //~| the size for values of type `Self` cannot be known
}

impl Trait2 for () {}

fn call_foo<T: Trait2>() {
    T::Associated::foo()
}

fn main() {
    call_foo::<()>()
}
