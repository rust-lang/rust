//! Check that we do not remove the storage statements if a reused local
//! is uninitialized in an unreachable block.
//@ test-mir-pass: GVN
// EMIT_MIR gvn_storage_unreachable.f.GVN.diff

#![feature(custom_mir, core_intrinsics)]

use std::intrinsics::mir::*;

struct S(u32, u32);

#[custom_mir(dialect = "runtime", phase = "post-cleanup")]
pub fn f(_1: u32) {
    // CHECK-LABEL: fn f(
    mir! {
        let _2: S;
        let _3: S;
        let _4: S;
        {
            // CHECK: StorageLive(_2);
            // CHECK: StorageLive(_3);
            // CHECK: _3 = copy _2;
            // CHECK: StorageDead(_3);
            // CHECK: StorageDead(_2);
            StorageLive(_2);
            _2 = S(_1, 2);
            StorageLive(_3);
            _3 = S(_1, 2);
            StorageDead(_3);
            StorageDead(_2);
            Return()
        }
        bb1 = {
            StorageLive(_2);
            // CHECK: _4 = copy _2;
            _4 = _2;
            StorageDead(_2);
            Return()
        }
    }
}
