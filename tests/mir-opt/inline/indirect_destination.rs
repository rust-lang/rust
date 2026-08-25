// Test for inlining with an indirect destination place.
//
//@ test-mir-pass: Inline
//@ edition: 2021
//@ needs-unwind
#![crate_type = "lib"]
#![feature(custom_mir, core_intrinsics)]
use core::intrinsics::mir::*;

#[custom_mir(dialect = "runtime", phase = "initial")]
// CHECK-LABEL: fn f(
// CHECK:      bb1: {
// CHECK-NEXT:   StorageLive([[A:.*]]);
// CHECK-NEXT:   [[A]] = &mut (*_1);
// CHECK-NEXT:   StorageLive([[B:.*]]);
// CHECK-NEXT:   [[B]] = const 42_u8;
// CHECK-NEXT:   (*[[A]]) = move [[B]];
// CHECK-NEXT:   StorageDead([[B]]);
// CHECK-NEXT:   StorageDead([[A]]);
// CHECK-NEXT:   goto -> bb1;
// CHECK-NEXT: }
pub fn f(a: *mut u8) {
    mir! {
        {
            Goto(bb1)
        }
        bb1 = {
            Call(*a = g(), ReturnTo(bb1), UnwindUnreachable())
        }
    }
}

#[custom_mir(dialect = "runtime", phase = "initial")]
#[inline(always)]
fn g() -> u8 {
    mir! {
        {
            RET = 42;
            Return()
        }
    }
}
