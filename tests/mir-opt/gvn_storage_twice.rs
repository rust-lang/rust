// skip-filecheck
//@ test-mir-pass: GVN
//@ compile-flags: -Zlint-mir=false

#![feature(custom_mir, core_intrinsics)]

use std::intrinsics::mir::*;

// EMIT_MIR gvn_storage_twice.repeat_local.GVN.diff
// EMIT_MIR gvn_storage_twice.repeat_local_dead.GVN.diff
// EMIT_MIR gvn_storage_twice.repeat_local_dead_live.GVN.diff

// Check that we remove the storage statements if the local
// doesn't have valid storage when it is used.
//
// Based on `gvn_repeat.rs::repeat_local`, were GVN should replace
// `let RET = *_5;` with `let RET = _3;`.

#[custom_mir(dialect = "runtime")]
pub fn repeat_local(_1: usize, _2: usize, _3: i32) -> i32 {
    mir! {
        {
            let _4 = [_3; 5];
            let _5 = &_4[_1];
            RET = *_5;
            Return()
        }
    }
}

// Since _3 is dead when we access _5, GVN should remove the storage statements.

#[custom_mir(dialect = "runtime")]
pub fn repeat_local_dead(_1: usize, _2: usize, _3: i32) -> i32 {
    mir! {
        {
            let _4 = [_3; 5];
            let _5 = &_4[_1];
            StorageDead(_3);
            RET = *_5;
            Return()
        }
    }
}

// Since _3 is uninitizaled when we access _5, GVN should _not_ optimze the code.

#[custom_mir(dialect = "runtime")]
pub fn repeat_local_dead_live(_1: usize, _2: usize, _3: i32) -> i32 {
    mir! {
        {
            let _4 = [_3; 5];
            let _5 = &_4[_1];
            StorageDead(_3);
            StorageLive(_3);
            RET = *_5;
            Return()
        }
    }
}

#[inline(never)]
fn opaque<T>(a: T) -> T {
    a
}
