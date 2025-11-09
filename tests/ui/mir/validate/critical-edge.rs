// Optimized MIR shouldn't have critical call edges
//
//@ edition: 2021
//@ compile-flags: -Zvalidate-mir
//@ failure-status: 101
//@ dont-check-compiler-stderr

#![feature(custom_mir, core_intrinsics)]
use std::intrinsics::mir::*;

#[custom_mir(dialect = "mono", phase = "optimized")]
#[inline(always)]
pub fn f(a: u32) -> u32 {
    mir! {
        {
            match a {
                0 => bb1,
                _ => bb2,
            }
        }
        bb1 = {
            Call(RET = f(1), ReturnTo(bb2), UnwindTerminate(ReasonAbi))
//~^ ERROR broken MIR in Item
//~| ERROR encountered critical edge in `Call` terminator
        }

        bb2 = {
            RET = 2;
            Return()
        }
    }
}

fn main() {
    f(5);
}
