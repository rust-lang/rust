// Checks to ensure that we properly detect when a closure constrains an existential type
#![feature(existential_type)]

use std::fmt::Debug;

fn main() {
    existential type Existential: Debug;
    fn _unused() -> Existential { String::new() }
    //~^ ERROR: concrete type differs from previous defining existential type use
    let null = || -> Existential { 0 };
    println!("{:?}", null());
}
