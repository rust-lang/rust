//! Ensure we detect aliasing of a in-place argument with the return place for the tricky case where
//! they do not live in memory.
//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
// Validation forces more things into memory, which we can't have here.
//@compile-flags: -Zmiri-disable-validation
#![feature(custom_mir, core_intrinsics)]
use std::intrinsics::mir::*;

#[allow(unused)]
pub struct S(i32);

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let _unit: ();
        {
            let staging = S(42); // This forces `staging` into memory...
            let _non_copy = staging; // ... so we move it to a non-inmemory local here.
            // This specifically uses a type with scalar representation to tempt Miri to use the
            // efficient way of storing local variables (outside adressable memory).
            Call(_non_copy = callee(Move(_non_copy)), ReturnTo(after_call), UnwindContinue())
            //~[stack]^ ERROR: not granting access
            //~[tree]| ERROR: /reborrow .* forbidden/
        }
        after_call = {
            Return()
        }
    }
}

pub fn callee(x: S) -> S {
    x
}
