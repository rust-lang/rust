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
    //~^ ERROR omitting type on const item declaration is experimental [E0658]
    //~| ERROR implemented const `K` has an incompatible type for trait [E0326]
    const Q = "";
    //~^ ERROR omitting type on const item declaration is experimental [E0658]
    //~| ERROR mismatched types [E0308]
    //~| ERROR lifetime parameters or bounds on associated constant `Q` do not match the trait declaration [E0195]
}

fn main() {}
