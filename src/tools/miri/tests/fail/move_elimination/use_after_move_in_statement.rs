//@revisions: normal move_elimination
//@[normal]check-pass
//@[move_elimination]compile-flags: -Zmir-move-elimination

#![feature(core_intrinsics, custom_mir)]

use std::intrinsics::mir::*;

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let value: (u8, u8);
        let ptr: *const (u8, u8);
        let result: ((u8, u8), u8);
        {
            value = (1, 2);
            ptr = &raw const value;
            result = (Move(value), (*ptr).0); //~[move_elimination] ERROR: has been freed
            Return()
        }
    }
}
