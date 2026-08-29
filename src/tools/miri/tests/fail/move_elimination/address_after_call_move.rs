//@revisions: normal move_elimination
//@[normal]check-pass
//@[move_elimination]compile-flags: -Zmir-move-elimination

#![feature(core_intrinsics, custom_mir)]

use std::intrinsics::mir::*;

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let value: (u8, u8);
        let unit: ();
        let ptr: *const (u8, u8);
        {
            value = (1, 2);
            Call(unit = consume(Move(value)), ReturnTo(after_call), UnwindContinue())
        }
        after_call = {
            ptr = &raw const value; //~[move_elimination] ERROR: live but unallocated
            Return()
        }
    }
}

fn consume(_: (u8, u8)) {}
