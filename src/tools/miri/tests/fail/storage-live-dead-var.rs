#![feature(core_intrinsics, custom_mir)]
use std::intrinsics::mir::*;

#[custom_mir(dialect = "runtime")]
fn main() {
    mir! {
        let val: i32;
        {
            val = 42; //~ERROR: accessing a dead local variable
            StorageLive(val); // too late... (but needs to be here to make `val` not implicitly live)
            Return()
        }
    }
}
