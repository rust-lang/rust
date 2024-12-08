//@ compile-flags: -Zlint-mir -Ztreat-err-as-bug
//@ build-fail
//@ failure-status: 101
//@ dont-check-compiler-stderr
//@ error-pattern: encountered overlapping memory in `Move` arguments to `Call`
#![feature(custom_mir, core_intrinsics)]
extern crate core;
use core::intrinsics::mir::*;

#[custom_mir(dialect = "runtime", phase = "optimized")]
pub fn main() {
    mir! {
        let a: [u8; 1024];
        {
            Call(a = f(Move(a)), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            Return()
        }
    }
}

pub fn f<T: Copy>(a: T) -> T { a }
