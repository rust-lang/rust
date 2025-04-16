//@ compile-flags: --crate-type=lib -Zlint-mir -Ztreat-err-as-bug
//@ build-fail
//@ failure-status: 101
//@ dont-check-compiler-stderr
//@ error-pattern: encountered `Assign` statement with overlapping memory
#![feature(custom_mir, core_intrinsics)]
extern crate core;
use core::intrinsics::mir::*;

#[custom_mir(dialect = "runtime", phase = "optimized")]
pub fn main() {
    mir! {
        let a: [u8; 1024];
        {
            a = a; //~ ERROR broken MIR
            Return()
        }
    }
}
