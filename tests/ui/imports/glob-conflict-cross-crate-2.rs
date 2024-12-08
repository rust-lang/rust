//@ aux-build:glob-conflict-cross-crate-2-extern.rs

extern crate glob_conflict_cross_crate_2_extern;

use glob_conflict_cross_crate_2_extern::*;

fn main() {
    let _a: C = 1; //~ ERROR cannot find type `C` in this scope
    //^ FIXME: `C` should be identified as an ambiguous item.
}
