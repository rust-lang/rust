//! Check that we do not remove storage statements when the head is alive for all usages.
//@ test-mir-pass: CopyProp
// EMIT_MIR issue_141649.f_move.CopyProp.diff
// EMIT_MIR issue_141649.f_head_borrowed.CopyProp.diff

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
            // CHECK: StorageLive(_1);
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
            // CHECK: StorageDead(_1);
            StorageDead(_2);
            StorageDead(_1);
            Return()
        }
    }
}

#[custom_mir(dialect = "runtime")]
fn f_head_borrowed() {
    // CHECK-LABEL: fn f_head_borrowed(
    mir! {
        let _1: S;
        let _2: S;
        let _3: S;
        let _4: &S;
        let _5: &S;
        {
            // CHECK: StorageLive(_1);
            // CHECK-NOT: StorageLive(_2);
            // CHECK: _3 = opaque::<S>(copy _1) -> [return: bb1, unwind unreachable];
            StorageLive(_1);
            _1 = S(1, 2);
            StorageLive(_2);
            _4 = &_1;
            _2 = _1;
            Call(_3 = opaque(Move(_1)), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            // CHECK-NOT: StorageDead(_2);
            // CHECK: StorageDead(_1);
            StorageDead(_2);
            StorageDead(_1);
            Call(_5 = opaque(Move(_4)), ReturnTo(bb2), UnwindUnreachable())
        }
        bb2 = {
            Return()
        }
    }
}

#[inline(never)]
fn opaque<T>(a: T) -> T {
    a
}
