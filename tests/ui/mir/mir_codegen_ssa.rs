//@ build-pass
//@ compile-flags: --crate-type=lib
#![feature(custom_mir, core_intrinsics)]
use std::intrinsics::mir::*;

#[custom_mir(dialect = "runtime", phase = "optimized")]
pub fn f(a: u32) -> u32 {
    mir! {
        let x: u32;
        {
            // Previously code generation failed with ICE "use of .. before def ..." because the
            // definition of x was incorrectly identified as dominating the use of x located in the
            // same statement:
            x = x + a;
            RET = x;
            Return()
        }
    }
}
