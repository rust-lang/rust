#![allow(incomplete_features)]
#![feature(non_lifetime_binders)]

trait Trait<T: ?Sized> {}

fn produce() -> impl for<T> Trait<(), Assoc = impl Trait<T>> {
    //~^ ERROR associated type `Assoc` not found for `Trait`
    //~| ERROR associated type `Assoc` not found for `Trait`
    //~| ERROR the trait bound `{integer}: Trait<()>` is not satisfied
    //~| ERROR cannot capture late-bound type parameter in nested `impl Trait`
    16
}

fn main() {}
