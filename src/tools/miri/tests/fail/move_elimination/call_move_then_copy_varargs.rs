//@revisions: stack tree
//@compile-flags: -Zmir-move-elimination
//@[tree]compile-flags: -Zmiri-tree-borrows

// Variadic arguments must be consumed in order: moving the first argument
// frees its source before the second argument is copied through an alias.

#![feature(core_intrinsics, custom_mir)]
use std::intrinsics::mir::*;

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let ptr: *const u64;
        let unit: ();
        {
            let value = 1u64;
            ptr = &raw const value;
            Call(unit = consume(Move(value), *ptr), ReturnTo(done), UnwindContinue())
        }
        done = { Return() }
    }
}

unsafe extern "C" fn consume(_: ...) {
    //~^ ERROR: has been freed
}
