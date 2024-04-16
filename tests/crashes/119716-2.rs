//@ known-bug: #119716
#![feature(non_lifetime_binders)]
trait Trait<T> {}
fn f<T>() -> impl for<T> Trait<impl Trait<T>> {}
