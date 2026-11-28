// This test features two examples of irreducible control flow in MIR, causing the main loop in
// `codegen_mir` (the one containing the calls to `codegen_block`) to iterate more than twice.
//
//@ build-pass
//@ compile-flags: -Zvalidate-mir
//@ edition: 2024

#![feature(core_intrinsics, custom_mir)]
#![allow(internal_features)]
#![crate_type = "lib"]

use std::intrinsics::mir::*;

use std::future::{Future, pending};
use std::pin::pin;
use std::task::Context;

// At the time of writing, this code results in irreducible MIR. That depends on the details of MIR
// generation, so we also have custom MIR functions below that don't depend on MIR generation.
async fn task() {
    loop {
        if std::hint::black_box(true) {
            pending::<()>().await;
            if const { std::mem::size_of::<u8>() } == 0 {
            } else {
                continue;
            }
        }
        std::hint::black_box(());
    }
}

pub fn repro1(cx: &mut Context<'_>) {
    let mut future = pin!(task());
    let _ = future.as_mut().poll(cx);
}

// `repro2` and `repro3` only differ in their value for `cond`. One of them will require more than
// two iterations; which one depends on the order in which traversal occurs. (At the time of writing
// it is `repro2`.)
#[custom_mir(dialect = "runtime", phase = "optimized")]
pub fn repro2() {
    mir! {
        let cond: bool;
        {
            cond = false;
            RET = (); // needed to prevent some folding that would neuter the test
            match cond {
                false => b,
                _ => a,
            }
        }
        a = {
            Goto(b)
        }
        b = {
            Goto(a)
        }
    }
}

// See the comments on `repro2`.
#[custom_mir(dialect = "runtime", phase = "optimized")]
pub fn repro3() {
    mir! {
        let cond: bool;
        {
            cond = true;
            RET = ();
            match cond {
                false => b,
                _ => a,
            }
        }
        a = {
            Goto(b)
        }
        b = {
            Goto(a)
        }
    }
}
