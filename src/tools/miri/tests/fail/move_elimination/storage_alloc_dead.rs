//@compile-flags: -Zmir-move-elimination

#![feature(core_intrinsics, custom_mir)]
use std::intrinsics::mir::*;

// StorageAlloc cannot start a storage lifetime.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let value: u32;
        {
            StorageAlloc(value); //~ ERROR: accessing a dead local variable
            StorageLive(value);
            Return()
        }
    }
}
