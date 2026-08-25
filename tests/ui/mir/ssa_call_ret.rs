// Regression test for issue #117331, where variable `a` was misidentified as
// being in SSA form (the definition occurs on the return edge only).
//
//@ edition:2021
//@ compile-flags: --crate-type=lib
//@ build-pass
//@ needs-unwind
#![feature(custom_mir, core_intrinsics)]
use core::intrinsics::mir::*;

#[custom_mir(dialect = "runtime", phase = "optimized")]
pub fn f() -> u32 {
    mir! {
        let a: u32;
        {
            Call(a = g(), ReturnTo(bb1), UnwindCleanup(bb2))
        }
        bb1 = {
            RET = a;
            Return()
        }
        bb2 (cleanup) = {
            RET = a;
            UnwindResume()
        }
    }
}

#[inline(never)]
pub fn g() -> u32 { 0 }
