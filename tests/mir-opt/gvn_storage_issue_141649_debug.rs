//! In lower opt levels, we remove any storage statements of reused locals.
//@ test-mir-pass: GVN
//@ compile-flags: -Copt-level=0
// EMIT_MIR gvn_storage_issue_141649_debug.f.GVN.diff

#![feature(custom_mir, core_intrinsics)]

use std::intrinsics::mir::*;

struct S(u32, u32);

#[custom_mir(dialect = "runtime")]
pub fn f(_1: u32) {
    // CHECK-LABEL: fn f(
    mir! {
        let _2: S;
        let _3: S;
        {
            // CHECK-NOT: StorageLive(_2);
            // CHECK: _3 = copy _2;
            // CHECK-NOT: StorageDead(_2);
            StorageLive(_2);
            _2 = S(_1, 2);
            StorageLive(_3);
            _3 = S(_1, 2);
            StorageDead(_3);
            StorageDead(_2);
            Return()
        }
    }
}
