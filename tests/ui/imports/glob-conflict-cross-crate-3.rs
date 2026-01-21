//@ aux-build:glob-conflict-cross-crate-2-extern.rs
//@ check-pass
extern crate glob_conflict_cross_crate_2_extern;

mod a {
    pub type C = i32;
}

use glob_conflict_cross_crate_2_extern::*;
use a::*;

fn main() {
    let _a: C = 1;
    //~^ WARN `C` is ambiguous
    //~| WARN `C` is ambiguous
    //~| WARN this was previously accepted
    //~| WARN this was previously accepted
}
