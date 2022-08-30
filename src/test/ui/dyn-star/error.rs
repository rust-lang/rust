#![feature(dyn_star)]

use std::fmt::Debug;

trait Foo {}

fn make_dyn_star() {
    let i = 42;
    let dyn_i: dyn* Foo = i as dyn* Foo; //~ ERROR trait bound `{integer}: Foo` is not satisfied
}

fn main() {}
