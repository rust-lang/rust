// Check that validation rejects moving a non-local, non-box place as a
// `Call` argument.
//
//@ failure-status: 101
//@ dont-check-compiler-stderr
//@ compile-flags: -Zvalidate-mir

#![feature(custom_mir, core_intrinsics)]
extern crate core;
use core::intrinsics::mir::*;

fn bar(_x: i32) {}

#[custom_mir(dialect = "built")]
pub fn main() {
    mir! {
        let a: (i32, i32);
        {
            a = (1, 2);
            Call(RET = bar(Move(a.0)), ReturnTo(retblock), UnwindContinue())
            //~^ ERROR broken MIR in
            //~| ERROR encountered `Move` of a non-local, non-box place in `Call` terminator
        }
        retblock = {
            Return()
        }
    }
}
