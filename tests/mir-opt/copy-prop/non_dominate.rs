// unit-test: CopyProp

#![feature(custom_mir, core_intrinsics)]
#![allow(unused_assignments)]
extern crate core;
use core::intrinsics::mir::*;

#[custom_mir(dialect = "analysis", phase = "post-cleanup")]
fn f(c: bool) -> bool {
    mir!(
        let a: bool;
        let b: bool;
        { Goto(bb1) }
        bb1 = { b = c; match b { false => bb3, _ => bb2 }}
        // This assignment to `a` does not dominate the use in `bb3`.
        // It should not be replaced by `b`.
        bb2 = { a = b; c = false; Goto(bb1) }
        bb3 = { RET = a; Return() }
    )
}

fn main() {
    assert_eq!(true, f(true));
}

// EMIT_MIR non_dominate.f.CopyProp.diff
