#![allow(incomplete_features)]
#![feature(non_lifetime_binders)]

type T = dyn for<V = A(&())> Fn(());
//~^ ERROR defaults for generic parameters are not allowed in `for<...>` binders
//~| ERROR cannot find type `A` in this scope
//~| ERROR late-bound type parameter not allowed on trait object types

fn main() {}
