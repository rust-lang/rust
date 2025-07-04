//@ test-mir-pass: CopyProp
//@ compile-flags: -Zlint-mir=false

#![feature(custom_mir, core_intrinsics)]
use std::intrinsics::mir::*;

// EMIT_MIR copy_prop_storage_preserve_head.f.CopyProp.diff
// EMIT_MIR copy_prop_storage_preserve_head.g.CopyProp.diff

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

#[custom_mir(dialect = "runtime")]
pub fn g() -> usize {
    // CHECK-LABEL: fn g(
    mir! {
        let _1: usize;
        let _2: usize;
        let _3: usize;
        // CHECK: bb0: {
        {
            // CHECK: StorageLive(_1);
            // CHECK: _0 = Add(copy _1, copy _1);
            // CHECK: StorageDead(_1);
            StorageLive(_2);
            StorageLive(_1);
            _1 = 0;
            _2 = _1;
            _3 = _2;
            RET = _3 + _3;
            // Even though the storage statements are in reverse order,
            // we should be able to keep the ones for _1.
            StorageDead(_1);
            StorageDead(_2);
            Return()
        }
    }
}
