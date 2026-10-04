//@revisions: normal move_elimination
//@[normal]check-pass
//@[move_elimination]compile-flags: -Zmir-move-elimination

#![feature(core_intrinsics, custom_mir)]
use std::intrinsics::mir::*;

// A later argument cannot read the pointer local consumed by an earlier move.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let ptr: *const u32;
        let unit: ();
        {
            let value = 42u32;
            ptr = &raw const value;
            Call(unit = consume(Move(ptr), *ptr), ReturnTo(done), UnwindContinue())
            //~[move_elimination]^ ERROR: accessing a live but unallocated local variable
        }
        done = { Return() }
    }
}
fn consume(_: *const u32, _: u32) {}
