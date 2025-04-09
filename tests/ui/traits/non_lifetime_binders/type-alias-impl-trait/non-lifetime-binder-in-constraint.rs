#![allow(incomplete_features)]
#![feature(non_lifetime_binders)]

trait Trait<T: ?Sized> {}

fn produce() -> impl for<T> Trait<(), Assoc = impl Trait<T>> {
    //~^ ERROR associated type `Assoc` not found for `Trait`
    //~| ERROR associated type `Assoc` not found for `Trait`
    //~| the trait bound `{integer}: Trait<()>` is not satisfied
    16
}

fn main() {}
