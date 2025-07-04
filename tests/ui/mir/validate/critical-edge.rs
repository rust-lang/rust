// Optimized MIR shouldn't have critical call edges
//
//@ build-fail
//@ edition: 2021
//@ compile-flags: --crate-type=lib -Zvalidate-mir
//@ failure-status: 101
//@ dont-check-compiler-stderr

#![feature(custom_mir, core_intrinsics)]
use core::intrinsics::mir::*;

#[custom_mir(dialect = "runtime", phase = "monomorphic")]
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
        }

        bb2 = {
            RET = 2;
            Return()
        }
    }
}

//~? RAW encountered critical edge in `Call` terminator
