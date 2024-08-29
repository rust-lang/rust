//@ compile-flags: --crate-type=lib
//@ edition:2021
#![feature(custom_mir, core_intrinsics)]
use core::intrinsics::mir::*;

// CHECK-LABEL: fn f()
// CHECK:       bb1 (cleanup): {
// CHECK-NEXT:  terminate(abi);
#[custom_mir(dialect = "runtime", phase = "optimized")]
pub fn f() {
    mir! {
        {
            Return()
        }
        bb1(cleanup) = {
            UnwindTerminate(ReasonAbi)
        }
    }
}

// CHECK-LABEL: fn g()
// CHECK:       bb1 (cleanup): {
// CHECK-NEXT:  terminate(cleanup);
#[custom_mir(dialect = "runtime", phase = "optimized")]
pub fn g() {
    mir! {
        {
            Return()
        }
        bb1(cleanup) = {
            UnwindTerminate(ReasonInCleanup)
        }
    }
}
