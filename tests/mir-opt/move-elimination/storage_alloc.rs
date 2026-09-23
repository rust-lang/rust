//@ test-mir-pass: MoveElimination
//@ compile-flags: -Cpanic=abort -Zmir-move-elimination

#![feature(core_intrinsics, custom_mir)]
#![allow(internal_features)]

use std::intrinsics::mir::*;

// The allocation starts the lifetime before the address is taken, and follows
// the local when it is moved into a field of the return place.
// EMIT_MIR storage_alloc.projected.MoveElimination.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn projected() -> ([u32; 4], u32) {
    // CHECK-LABEL: fn projected(
    // CHECK: StorageAlloc(_0);
    // CHECK: &raw mut (_0.0: [u32; 4]);
    mir! {
        let value: [u32; 4];
        let ptr: *mut [u32; 4];
        {
            StorageLive(value);
            StorageAlloc(value);
            ptr = &raw mut value;
            value = [42_u32; 4];
            RET = (Move(value), 99_u32);
            StorageDead(value);
            Return()
        }
    }
}

fn main() {
    assert_eq!(projected(), ([42; 4], 99));
}
