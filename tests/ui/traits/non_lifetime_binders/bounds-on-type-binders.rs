//@ check-pass

#![allow(incomplete_features)]
#![feature(non_lifetime_binders)]

trait Trait {}

trait Trait2
where
    for<T: Trait> ():,
{
}

fn main() {}
