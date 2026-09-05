//@revisions: normal move_elimination
//@[normal]check-pass
//@[move_elimination]compile-flags: -Zmir-move-elimination

#![feature(core_intrinsics, custom_mir)]

use std::intrinsics::mir::*;

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let ptr: *mut bool;
        let value: bool;
        {
            ptr = &raw mut value; //~[move_elimination] ERROR: live but unallocated
            *ptr = true;
            Return()
        }
    }
}
