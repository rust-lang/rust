#![allow(incomplete_features)]
#![feature(non_lifetime_binders)]

trait Trait<T> {}

fn f() -> impl for<T> Trait<impl Trait<T>> {}
//~^ ERROR nested `impl Trait` is not allowed
//~| ERROR the trait bound `(): Trait<impl Trait<T>>` is not satisfied

fn main() {}
