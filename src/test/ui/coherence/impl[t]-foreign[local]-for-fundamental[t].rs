// compile-flags:--crate-name=test
// aux-build:coherence_lib.rs

extern crate coherence_lib as lib;
use lib::*;
use std::rc::Rc;

struct Local;

impl<T> Remote1<Local> for Box<T> {
    //~^ ERROR type parameter `T` as argument to a fundamental type
    // | must be covered by another type when it appears before the first local type (`Local`)
}

impl<T> Remote1<Local> for &T {
    //~^ ERROR type parameter `T` as argument to a fundamental type
    // | must be covered by another type when it appears before the first local type (`Local`)
}

fn main() {}
