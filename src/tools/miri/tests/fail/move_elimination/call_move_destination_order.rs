//@revisions: normal move_elimination
//@[normal]check-pass
//@[move_elimination]compile-flags: -Zmir-move-elimination

#![feature(core_intrinsics, custom_mir)]
use std::intrinsics::mir::*;

// Destination evaluation cannot read a pointer local moved by an argument.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let ptr: *mut u32;
        {
            let value = 0u32;
            ptr = &raw mut value;
            Call(*ptr = consume(Move(ptr)), ReturnTo(done), UnwindContinue())
            //~[move_elimination]^ ERROR: accessing a live but unallocated local variable
        }
        done = { Return() }
    }
}
fn consume(_: *mut u32) -> u32 {
    42
}
