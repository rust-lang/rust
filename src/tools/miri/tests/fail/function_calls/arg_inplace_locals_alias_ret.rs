//! Ensure we detect aliasing of an in-place argument with the return place for the tricky case where
//! they do not live in memory. With move elimination, return-place protection must detect
//! that argument passing has freed the destination allocation.
//@revisions: stack tree stack_move_elimination tree_move_elimination
//@[tree]compile-flags: -Zmiri-tree-borrows
//@[stack_move_elimination]compile-flags: -Zmir-move-elimination
//@[tree_move_elimination]compile-flags: -Zmiri-tree-borrows -Zmir-move-elimination
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
            // efficient way of storing local variables (outside addressable memory).
            Call(_non_copy = callee(Move(_non_copy)), ReturnTo(after_call), UnwindContinue())
        }
        after_call = {
            Return()
        }
    }
}

fn callee(x: S) -> S {
    //~[stack]^ ERROR: not granting access
    //~[tree]| ERROR: /reborrow .* forbidden/
    //~[stack_move_elimination]| ERROR: has been freed
    //~[tree_move_elimination]| ERROR: has been freed
    x
}
