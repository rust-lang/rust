//@compile-flags: -Zmir-move-elimination

#![feature(core_intrinsics, custom_mir)]
use std::intrinsics::mir::*;

// Returning from a function requires its return local to be allocated.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn uninitialized() -> u32 {
    mir! {
        {
            Return() //~ ERROR: accessing a live but unallocated local variable
        }
    }
}

fn main() {
    let _ = uninitialized();
}
