// compile-flags:--crate-name=test
// aux-build:coherence_lib.rs

extern crate coherence_lib as lib;
use lib::*;
use std::rc::Rc;

struct Local;

impl<T> Remote for Box<T> {
    //~^ ERROR type parameter `T` as argument to a fundamental type must be used as the type parameter for
    // | some local type (e.g., `MyStruct<T>`)
}

fn main() {}
