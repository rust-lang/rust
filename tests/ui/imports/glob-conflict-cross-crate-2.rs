// check-pass
// aux-build:glob-conflict-cross-crate-2-extern.rs

extern crate glob_conflict_cross_crate_2_extern;

use glob_conflict_cross_crate_2_extern::*;

fn main() {
    let _a: C = 1;
    //~^ WARNING: `C` is ambiguous
    //~| WARNING: this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!
}
