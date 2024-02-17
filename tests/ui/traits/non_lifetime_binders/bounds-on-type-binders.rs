//@ check-fail

#![allow(incomplete_features)]
#![feature(non_lifetime_binders)]

trait Trait {}

trait Trait2
where
    for<T: Trait> ():,
{ //~^ ERROR bounds cannot be used in this context
}

fn main() {}
