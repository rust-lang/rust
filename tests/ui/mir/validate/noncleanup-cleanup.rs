// Check that validation rejects cleanup edge to a non-cleanup block.
//
//@ failure-status: 101
//@ dont-check-compiler-stderr
//@ error-pattern: cleanuppad mismatch
#![feature(custom_mir, core_intrinsics)]
extern crate core;
use core::intrinsics::mir::*;

#[custom_mir(dialect = "built")]
pub fn main() {
    mir! {
        {
            Call(RET = main(), ReturnTo(block), UnwindCleanup(block))
        }
        block = {
            Return()
        }
    }
}
