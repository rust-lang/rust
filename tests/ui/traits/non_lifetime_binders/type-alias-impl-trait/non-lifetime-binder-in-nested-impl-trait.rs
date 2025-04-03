// compile-flags: --edition=2021
#![feature(non_lifetime_binders)]
#![feature(associated_type_defaults)]
#![allow(incomplete_features)]

trait Trait<T: ?Sized> {
    type Assoc<'a> = i32;
}

fn produce() -> impl for<T> Trait<(), Assoc = impl Trait<T>> {
    //~^ ERROR missing generics for associated type
    //~| ERROR the trait bound
    //~| ERROR missing generics for associated type
    //~| note: associated type defined here
    //~| help: add missing lifetime argument
    //~| note: duplicate diagnostic
    //~| help: add missing lifetime argument
    16
}

fn main() {}
