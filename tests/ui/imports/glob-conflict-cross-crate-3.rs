//@ check-pass
//@ aux-build:glob-conflict-cross-crate-2-extern.rs

extern crate glob_conflict_cross_crate_2_extern;

mod a {
    pub type C = i32;
}

use glob_conflict_cross_crate_2_extern::*;
use a::*;

fn main() {
    let _a: C = 1;
    //^ FIXME: `C` should be identified as an ambiguous item.
}
