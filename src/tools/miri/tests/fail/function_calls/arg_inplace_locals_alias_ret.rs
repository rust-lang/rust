//! Ensure we detect aliasing of an in-place argument with the return place for the tricky case where
//! they do not live in memory.
//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
#![feature(custom_mir, core_intrinsics)]
use std::intrinsics::mir::*;

#[allow(unused)]
#[repr(transparent)]
pub struct S(i32);

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let _unit: ();
        {
            let staging = S(42); // This forces `staging` into memory...
            let _non_copy = staging; // ... so we move it to a non-inmemory local here.
            // This specifically uses a type with scalar representation to tempt Miri to use the
            // efficient way of storing local variables (outside addressable memory).
            // Also use a projection otherwise this would be considered malformed MIR.
            Call(_non_copy = callee(Move(_non_copy.0)), ReturnTo(after_call), UnwindContinue())
        }
        after_call = {
            Return()
        }
    }
}

fn callee(x: i32) -> S {
    //~[stack]^ ERROR: not granting access
    //~[tree]| ERROR: /reborrow .* forbidden/
    S(x)
}
