//@ compile-flags: -Zlint-mir -Ztreat-err-as-bug -Zeagerly-emit-delayed-bugs
//@ failure-status: 101
//@ dont-check-compiler-stderr
//@ error-pattern: has storage when returning
#![feature(custom_mir, core_intrinsics)]
extern crate core;
use core::intrinsics::mir::*;

#[custom_mir(dialect = "built")]
fn main() {
    mir! {
        let a: ();
        {
            StorageLive(a);
            RET = a;
            Return() //~ ERROR broken MIR
        }
    }
}
