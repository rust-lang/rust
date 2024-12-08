//@ check-pass

#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

trait Trait<T: ?Sized> {}

impl<T: ?Sized> Trait<T> for i32 {}

fn produce() -> impl for<T> Trait<T> {
    16
}

fn main() {
    let _ = produce();
}
