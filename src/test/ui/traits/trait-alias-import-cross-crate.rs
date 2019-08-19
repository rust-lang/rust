// run-pass
// aux-build:trait_alias.rs

#![feature(trait_alias)]

extern crate trait_alias;

// Import only the alias, not the real trait.
use trait_alias::{Greet, Hi};

fn main() {
    let hi = Hi;
    hi.hello(); // From `Hello`, via `Greet` alias.
}
