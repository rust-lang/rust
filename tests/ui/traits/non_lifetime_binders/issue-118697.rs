#![allow(incomplete_features)]
#![feature(non_lifetime_binders)]

type T = dyn for<V = A(&())> Fn(());
//~^ ERROR default parameter is not allowed in this binder
//~| ERROR cannot find type `A` in this scope
//~| ERROR late-bound type parameter not allowed on trait object types

fn main() {}
