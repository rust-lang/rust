// `SetDiscriminant` does not actually write anything if the chosen variant is the untagged variant
// of a niche encoding. Verify that we do not thread over this case.
//@ unit-test: JumpThreading

#![feature(custom_mir)]
#![feature(core_intrinsics)]

use std::intrinsics::mir::*;

enum E<T> {
    A,
    B(T),
}

// EMIT_MIR set_no_discriminant.f.JumpThreading.diff
#[custom_mir(dialect = "runtime")]
pub fn f() -> usize {
    // CHECK-LABEL: fn f(
    // CHECK-NOT: goto
    // CHECK: switchInt(
    // CHECK-NOT: goto
    mir!(
        let a: isize;
        let e: E<char>;
        {
            e = E::A;
            SetDiscriminant(e, 1);
            a = Discriminant(e);
            match a {
                0 => bb0,
                _ => bb1,
            }
        }
        bb0 = {
            RET = 0;
            Return()
        }
        bb1 = {
            RET = 1;
            Return()
        }
    )
}

// EMIT_MIR set_no_discriminant.generic.JumpThreading.diff
#[custom_mir(dialect = "runtime")]
pub fn generic<T>() -> usize {
    // CHECK-LABEL: fn generic(
    // CHECK-NOT: goto
    // CHECK: switchInt(
    // CHECK-NOT: goto
    mir!(
        let a: isize;
        let e: E<T>;
        {
            e = E::A;
            SetDiscriminant(e, 1);
            a = Discriminant(e);
            match a {
                0 => bb0,
                _ => bb1,
            }
        }
        bb0 = {
            RET = 0;
            Return()
        }
        bb1 = {
            RET = 1;
            Return()
        }
    )
}

fn main() {
    assert_eq!(f(), 0);
    assert_eq!(generic::<char>(), 0);
}
