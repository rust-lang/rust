//@ test-mir-pass: CopyProp
//@ compile-flags: -Zlint-mir=false

#![feature(custom_mir, core_intrinsics)]
use std::intrinsics::mir::*;

// EMIT_MIR copy_prop_storage_preserve_head.f.CopyProp.diff

#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn f(_1: &mut usize) {
    // CHECK-LABEL: fn f(
    mir! {
        let _2: usize;
        let _3: usize;
        // CHECK: bb0: {
        {
            // CHECK: StorageLive(_2);
            // CHECK: (*_1) = copy _2;
            // CHECK: StorageDead(_2);
            StorageLive(_2);
            _2 = 0;
            _3 = _2;
            (*_1) = _3;
            StorageDead(_2);
            (*_1) = _2;
            Return()
        }
    }
}
