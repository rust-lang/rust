// skip-filecheck
//@ test-mir-pass: CopyProp

#![feature(custom_mir, core_intrinsics, freeze)]

// Check that we remove the storage statements if one of the locals is borrowed,
// and the head isn't borrowed.

use std::intrinsics::mir::*;
use std::marker::Freeze;

// EMIT_MIR copy_prop_borrowed_storage_not_removed.f.CopyProp.diff

#[custom_mir(dialect = "runtime")]
pub fn f<T: Copy + Freeze>(_1: (T, T)) -> T {
    mir! {
        let _2: T;
        let _3: T;
        let _4: &T;
        {
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
