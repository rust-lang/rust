// `SetDiscriminant` does not actually write anything if the chosen variant is the untagged variant
// of a niche encoding. However, it is UB to call `SetDiscriminant` with the untagged variant if the
// value currently encodes a different variant. Verify that we do correctly thread in this case.
//@ test-mir-pass: JumpThreading

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
    // CHECK-NOT: switchInt
    // CHECK: goto
    // CHECK-NOT: switchInt
    mir! {
        let a: isize;
        let e: E<char>;
        {
            e = E::A;
            SetDiscriminant(e, 1); // UB!
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
    }
}

// EMIT_MIR set_no_discriminant.generic.JumpThreading.diff
#[custom_mir(dialect = "runtime")]
pub fn generic<T>() -> usize {
    // CHECK-LABEL: fn generic(
    // CHECK-NOT: switchInt
    // CHECK: goto
    // CHECK-NOT: switchInt
    mir! {
        let a: isize;
        let e: E<T>;
        {
            e = E::A;
            SetDiscriminant(e, 1); // UB!
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
    }
}

// CHECK-LABEL: fn main(
fn main() {
    assert_eq!(f(), 0);
    assert_eq!(generic::<char>(), 0);
}
