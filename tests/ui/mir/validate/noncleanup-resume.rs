// Check that validation rejects resume terminator in a non-cleanup block.
//
//@ failure-status: 101
//@ dont-check-compiler-stderr

#![feature(custom_mir, core_intrinsics)]
extern crate core;
use core::intrinsics::mir::*;

#[custom_mir(dialect = "built")]
pub fn main() {
    mir! {
        {
            UnwindResume() //~ ERROR resume on non-cleanup block
        }
    }
}
