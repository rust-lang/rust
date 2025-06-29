//! In lower opt levels, we remove (more) storage statements using a simpler strategy.
//@ test-mir-pass: CopyProp
//@ compile-flags: -Copt-level=0
// EMIT_MIR issue_141649_debug.f_move.CopyProp.diff

#![feature(custom_mir, core_intrinsics)]

use std::intrinsics::mir::*;

struct S(u32, u32);

#[custom_mir(dialect = "runtime")]
pub fn f_move() {
    // CHECK-LABEL: fn f_move(
    mir! {
        let _1: S;
        let _2: S;
        let _3: S;
        {
            // CHECK-NOT: StorageLive(_1);
            // CHECK-NOT: StorageLive(_2);
            // CHECK: _3 = opaque::<S>(copy _1) -> [return: bb1, unwind unreachable];
            StorageLive(_1);
            _1 = S(1, 2);
            StorageLive(_2);
            _2 = _1;
            Call(_3 = opaque(Move(_1)), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            // CHECK-NOT: StorageDead(_2);
            // CHECK-NOT: StorageDead(_1);
            StorageDead(_2);
            StorageDead(_1);
            Return()
        }
    }
}

#[inline(never)]
fn opaque<T>(a: T) -> T {
    a
}
