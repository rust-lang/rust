// Test that we do not move multiple times from the same local.
//@ test-mir-pass: CopyProp
//@ compile-flags: --crate-type=lib -Cpanic=abort
#![feature(custom_mir, core_intrinsics)]
use core::intrinsics::mir::*;
use core::mem::MaybeUninit;
extern crate core;

#[custom_mir(dialect = "runtime", phase = "initial")]
pub fn moved_and_copied<T: Copy>(_1: T) {
    // CHECK-LABEL: fn moved_and_copied(
    // CHECK: _0 = f::<T>(copy _1, copy _1)
    mir! {
        {
            let _2 = _1;
            Call(RET = f(Move(_1), Move(_2)), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            Return()
        }
    }
}

#[custom_mir(dialect = "runtime", phase = "initial")]
pub fn moved_twice<T: Copy>(_1: MaybeUninit<T>) {
    // In a future we would like to propagate moves instead of copies here. The resulting program
    // would have an undefined behavior due to overlap in a call terminator, so we need to change
    // operational semantics to explain why the original program has undefined behavior.
    // https://github.com/rust-lang/unsafe-code-guidelines/issues/556
    //
    // CHECK-LABEL: fn moved_twice(
    // CHECK: _0 = f::<MaybeUninit<T>>(copy _1, copy _1)
    mir! {
        {
            let _2 = Move(_1);
            let _3 = Move(_1);
            Call(RET = f(Move(_2), Move(_3)), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            Return()
        }
    }
}

#[inline(never)]
pub fn f<T: Copy>(_: T, _: T) {}
