//@ aux-build:glob-conflict-cross-crate-2-extern.rs
//@ check-pass
extern crate glob_conflict_cross_crate_2_extern;

use glob_conflict_cross_crate_2_extern::*;

fn main() {
    let _a: C = 1; //~ WARN `C` is ambiguous
                   //~| WARN this was previously accepted
}
