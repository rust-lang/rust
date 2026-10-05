//@revisions: normal move_elimination
//@[normal]check-pass
//@[move_elimination]compile-flags: -Zmir-move-elimination

#![feature(core_intrinsics, custom_mir)]
use std::intrinsics::mir::*;
use std::mem::MaybeUninit;

// Reject the unallocated argument before destination evaluation allocates its local.
// MaybeUninit permits uninitialized bytes, isolating the allocation-state check.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let value: MaybeUninit<u32>;
        {
            StorageLive(value);
            Call(value = identity(value), ReturnTo(done), UnwindContinue())
            //~[move_elimination]^ ERROR: accessing a live but unallocated local variable
        }
        done = { Return() }
    }
}

fn identity(value: MaybeUninit<u32>) -> MaybeUninit<u32> {
    value
}
