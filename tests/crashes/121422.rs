//@ known-bug: #121422
#![feature(non_lifetime_binders)]

trait Trait<T: ?Sized> {}

fn produce() -> impl for<T> Trait<(), Assoc = impl Trait<T>> {
    16
}
