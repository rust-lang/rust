//! Check that we remove the storage statements if one of the locals is borrowed,
//! and the head isn't borrowed.
//@ test-mir-pass: CopyProp

#![feature(custom_mir, core_intrinsics, freeze)]

use std::intrinsics::mir::*;
use std::marker::Freeze;

// EMIT_MIR copy_prop_storage_removed_when_local_borrowed.f.CopyProp.diff

#[custom_mir(dialect = "runtime")]
pub fn f<T: Copy + Freeze>(_1: (T, T)) -> T {
    // CHECK-LABEL: fn f(
    mir! {
        let _2: T;
        let _3: T;
        let _4: &T;
        // CHECK: bb0: {
        {
            // FIXME: Currently, copy propagation will not unify borrowed locals.
            // If it does, the storage statements for `_2` should be remove
            // so these checks will need to be updated.
            // CHECK: StorageLive(_2);
            // CHECK: _4 = &_3;
            // CHECK: StorageDead(_2);
            StorageLive(_2);
            _2 = _1.0;
            _3 = _2;
            _4 = &_3;
            StorageDead(_2);
            RET = *_4;
            Return()
        }
    }
}
