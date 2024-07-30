//@ known-bug: #119716
#![feature(non_lifetime_binders)]
trait Trait<T> {}
fn f() -> impl for<T> Trait<impl Trait<T>> {}
