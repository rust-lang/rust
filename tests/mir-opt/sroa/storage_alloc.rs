//@ test-mir-pass: ScalarReplacementOfAggregates

#![crate_type = "lib"]
#![feature(custom_mir, core_intrinsics)]
#![allow(internal_features)]

use std::intrinsics::mir::*;
use std::mem::MaybeUninit;

// Explicit StorageAlloc of an aggregate must allocate every replacement field.
// EMIT_MIR storage_alloc.fields.ScalarReplacementOfAggregates.diff
#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn fields() -> (MaybeUninit<u8>, MaybeUninit<u16>) {
    // CHECK-LABEL: fn fields(
    // CHECK: bb0: {
    // CHECK-NEXT: StorageAlloc([[FIRST:_[0-9]+]]);
    // CHECK-NEXT: StorageAlloc([[SECOND:_[0-9]+]]);
    // CHECK: _0 = (copy [[FIRST]], copy [[SECOND]]);
    mir! {
        let x: (MaybeUninit<u8>, MaybeUninit<u16>);
        {
            StorageAlloc(x);
            RET = (x.0, x.1);
            Return()
        }
    }
}
