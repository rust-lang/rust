//@ run-pass
//@ aux-build:greeter.rs

#![feature(trait_alias)]

extern crate greeter;

// Import only the alias, not the real trait.
use greeter::{Greet, Hi};

fn main() {
    let hi = Hi;
    hi.hello(); // From `Hello`, via `Greet` alias.
}
