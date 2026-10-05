//@revisions: normal stack tree
//@[stack]compile-flags: -Zmir-move-elimination
//@[tree]compile-flags: -Zmir-move-elimination -Zmiri-tree-borrows

#![feature(core_intrinsics, custom_mir)]
use std::intrinsics::mir::*;

// Allocate before taking an address, then preserve the value and pointer across
// another StorageAlloc. It must not reset storage or perform a conflicting access.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn allocated() -> u32 {
    mir! {
        let value: u32;
        let ptr: *mut u32;
        {
            StorageLive(value);
            StorageAlloc(value);
            ptr = &raw mut value;
            *ptr = 42;
            StorageAlloc(value);
            RET = *ptr;
            StorageDead(value);
            Return()
        }
    }
}

// Allocation must also preserve a value held as an immediate.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn immediate() -> u32 {
    mir! {
        {
            let value = 42_u32;
            StorageAlloc(value);
            RET = value;
            Return()
        }
    }
}

fn main() {
    assert_eq!(allocated(), 42);
    assert_eq!(immediate(), 42);
}
