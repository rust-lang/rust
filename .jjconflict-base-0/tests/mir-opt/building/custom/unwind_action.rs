//@ compile-flags: --crate-type=lib
//@ edition:2021
//@ needs-unwind
#![feature(custom_mir, core_intrinsics)]
use core::intrinsics::mir::*;

// CHECK-LABEL: fn a()
// CHECK:       bb0: {
// CHECK-NEXT:  a() -> [return: bb1, unwind unreachable];
#[custom_mir(dialect = "runtime", phase = "optimized")]
pub fn a() {
    mir! {
        {
            Call(RET = a(), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            Return()
        }
    }
}

// CHECK-LABEL: fn b()
// CHECK:       bb0: {
// CHECK-NEXT:  b() -> [return: bb1, unwind continue];
#[custom_mir(dialect = "runtime", phase = "optimized")]
pub fn b() {
    mir! {
        {
            Call(RET = b(), ReturnTo(bb1), UnwindContinue())
        }
        bb1 = {
            Return()
        }
    }
}

// CHECK-LABEL: fn c()
// CHECK:       bb0: {
// CHECK-NEXT:  c() -> [return: bb1, unwind terminate(abi)];
#[custom_mir(dialect = "runtime", phase = "optimized")]
pub fn c() {
    mir! {
        {
            Call(RET = c(), ReturnTo(bb1), UnwindTerminate(ReasonAbi))
        }
        bb1 = {
            Return()
        }
    }
}

// CHECK-LABEL: fn d()
// CHECK:       bb0: {
// CHECK-NEXT:  d() -> [return: bb1, unwind: bb2];
#[custom_mir(dialect = "runtime", phase = "optimized")]
pub fn d() {
    mir! {
        {
            Call(RET = d(), ReturnTo(bb1), UnwindCleanup(bb2))
        }
        bb1 = {
            Return()
        }
        bb2 (cleanup) = {
            UnwindResume()
        }
    }
}
