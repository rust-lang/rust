//@ run-pass
//@ aux-build:greeter.rs


extern crate greeter;

// Import only the alias, not the real trait.
use greeter::{Greet, Hi};

fn main() {
    let hi = Hi;
    hi.hello(); // From `Hello`, via `Greet` alias.
}
