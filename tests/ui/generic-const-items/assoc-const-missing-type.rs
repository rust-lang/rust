// Ensure that we properly deal with missing/placeholder types inside GACs.
// issue: rust-lang/rust#124833
#![feature(generic_const_items)]
#![allow(incomplete_features)]

trait Trait {
    const K<T>: T;
    const Q<'a>: &'a str;
}

impl Trait for () {
    const K<T> = ();
    //~^ ERROR missing type for `const` item
    //~| ERROR mismatched types
    const Q = "";
    //~^ ERROR missing type for `const` item
    //~| ERROR lifetime parameters or bounds on associated const `Q` do not match the trait declaration
}

fn main() {}
