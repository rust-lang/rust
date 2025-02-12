#![feature(dyn_star)]
#![allow(incomplete_features)]

use std::fmt::Debug;

trait Foo {}

fn make_dyn_star() {
    let i = 42usize;
    let dyn_i: dyn* Foo = i; //~ ERROR trait bound `usize: Foo` is not satisfied
}

fn main() {}
