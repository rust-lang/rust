//@ test-mir-pass: CopyProp

#![crate_type = "lib"]
#![feature(custom_mir, core_intrinsics)]
#![allow(internal_features)]

use std::intrinsics::mir::*;

// Ensure the copy source's storage lifetime covers the redirected StorageAlloc.
// EMIT_MIR storage_alloc.before_source.CopyProp.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn before_source() -> u8 {
    // CHECK-LABEL: fn before_source(
    // CHECK: bb0: {
    // CHECK-NOT: StorageLive
    // CHECK: StorageAlloc([[SOURCE:_[0-9]+]]);
    // CHECK-NOT: StorageLive
    // CHECK: [[SOURCE]] = const 42_u8;
    // CHECK-NOT: StorageDead
    // CHECK: return;
    mir! {
        let x: u8;
        let y: u8;
        {
            StorageLive(y);
            StorageAlloc(y);
            StorageLive(x);
            x = 42;
            y = x;
            RET = y + 1;
            StorageDead(x);
            StorageDead(y);
            Return()
        }
    }
}
