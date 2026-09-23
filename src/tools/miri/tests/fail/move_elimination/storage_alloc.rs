//@compile-flags: -Zmir-move-elimination

#![feature(core_intrinsics, custom_mir)]
use std::intrinsics::mir::*;

// StorageAlloc provides storage, but does not initialize the value.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let value: u32;
        let ptr: *const u32;
        let result: u32;
        {
            StorageAlloc(value);
            ptr = &raw const value;
            result = *ptr;
            //~^ ERROR: uninitialized
            Return()
        }
    }
}
