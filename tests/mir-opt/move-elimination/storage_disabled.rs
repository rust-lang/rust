//@ test-mir-pass: MoveElimination
//@ compile-flags: -Copt-level=0 -Cpanic=abort

#![feature(custom_mir, core_intrinsics)]

use std::intrinsics::mir::*;

// With storage reconstruction disabled, existing storage markers are removed
// without inserting new ones.
// EMIT_MIR storage_disabled.disabled.MoveElimination.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn disabled() -> *const u32 {
    // CHECK-LABEL: fn disabled(
    // CHECK: bb0: {
    // CHECK-NOT: Storage
    // CHECK: [[X:_[0-9]+]] = const 1_u32;
    // CHECK-NOT: Storage
    // CHECK: return;
    // CHECK-NOT: Storage
    mir! {
        let x: u32;
        {
            StorageLive(x);
            x = 1;
            RET = &raw const x;
            StorageDead(x);
            Return()
        }
    }
}
