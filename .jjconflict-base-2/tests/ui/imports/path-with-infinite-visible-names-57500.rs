// Regression test for <https://github.com/rust-lang/rust/issues/57500>.
// An item reachable under infinitely many paths used to hang path printing
// while rendering this error.
//@ aux-build: pathloop.rs

extern crate pathloop;

use pathloop::prelude::*;

fn main() {
    let _x: AStruct = 42; //~ ERROR mismatched types
}
