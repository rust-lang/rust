//@ test-mir-pass: CopyProp
//@ compile-flags: -Zlint-mir=false

#![feature(custom_mir, core_intrinsics)]

// Check that we remove the storage statements if the head
// becomes uninitialized before it is used again.

use std::intrinsics::mir::*;

// EMIT_MIR copy_prop_storage_twice.dead_twice.CopyProp.diff
#[custom_mir(dialect = "runtime")]
pub fn dead_twice<T: Copy>(_1: T) -> T {
    // CHECK-LABEL: fn dead_twice(
    mir! {
        let _2: T;
        let _3: T;
        {
            // CHECK-NOT: StorageLive(_2);
            StorageLive(_2);
            Call(_2 = opaque(Move(_1)), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            // CHECK-NOT: StorageDead(_2);
            // CHECK-NOT: StorageLive(_2);
            // CHECK: _0 = opaque::<T>(copy _2) -> [return: bb2, unwind unreachable];
            let _3 = Move(_2);
            StorageDead(_2);
            StorageLive(_2);
            Call(RET = opaque(Move(_3)), ReturnTo(bb2), UnwindUnreachable())
        }
        bb2 = {
            // CHECK-NOT: StorageDead(_2);
            StorageDead(_2);
            Return()
        }
    }
}

// EMIT_MIR copy_prop_storage_twice.live_twice.CopyProp.diff
#[custom_mir(dialect = "runtime")]
pub fn live_twice<T: Copy>(_1: T) -> T {
    // CHECK-LABEL: fn live_twice(
    mir! {
        let _2: T;
        let _3: T;
        {
            // CHECK-NOT: StorageLive(_2);
            StorageLive(_2);
            Call(_2 = opaque(Move(_1)), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            // CHECK-NOT: StorageLive(_2);
            // CHECK: _0 = opaque::<T>(copy _2) -> [return: bb2, unwind unreachable];
            let _3 = Move(_2);
            StorageLive(_2);
            Call(RET = opaque(_3), ReturnTo(bb2), UnwindUnreachable())
        }
        bb2 = {
            // CHECK-NOT: StorageDead(_2);
            StorageDead(_2);
            Return()
        }
    }
}

#[inline(never)]
fn opaque<T>(a: T) -> T {
    a
}
