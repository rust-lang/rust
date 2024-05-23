// Ensure that we properly deal with missing/placeholder types inside GACs.
// issue: rust-lang/rust#124833
#![feature(generic_const_items)]
#![allow(incomplete_features)]

trait Trait {
    const K<T>: T;
}

impl Trait for () {
    const K<T> = ();
    //~^ ERROR missing type for `const` item
    //~| ERROR mismatched types
    //~| ERROR mismatched types
}

fn main() {}
