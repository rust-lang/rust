//@ test-mir-pass: DeadStoreElimination-initial

#![crate_type = "lib"]
#![feature(custom_mir, core_intrinsics)]
#![allow(internal_features)]

use std::intrinsics::mir::*;
use std::mem::MaybeUninit;

// The assignment provides the allocation needed by the later read.
// EMIT_MIR storage_alloc.dead.DeadStoreElimination-initial.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn dead(value: MaybeUninit<u8>) -> MaybeUninit<u8> {
    // CHECK-LABEL: fn dead(
    // CHECK-NOT: StorageAlloc
    // CHECK: return;
    mir! {
        let x: MaybeUninit<u8>;
        {
            StorageAlloc(x);
            x = value;
            RET = x;
            Return()
        }
    }
}

// Uninitialized bytes may be read as MaybeUninit, but the allocation must exist.
// EMIT_MIR storage_alloc.live.DeadStoreElimination-initial.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn live() -> MaybeUninit<u8> {
    // CHECK-LABEL: fn live(
    // CHECK: StorageAlloc([[LOCAL:_[0-9]+]]);
    // CHECK-NEXT: _0 = copy [[LOCAL]];
    mir! {
        let x: MaybeUninit<u8>;
        {
            StorageAlloc(x);
            RET = x;
            Return()
        }
    }
}
