//! Ensure we detect aliasing of two in-place arguments for the tricky case where they do not
//! live in memory.
//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
// Validation forces more things into memory, which we can't have here.
//@compile-flags: -Zmiri-disable-validation
#![feature(custom_mir, core_intrinsics)]
use std::intrinsics::mir::*;

pub struct S(i32);

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let _unit: ();
        {
            let staging = S(42); // This forces `staging` into memory...
            let non_copy = staging; // ... so we move it to a non-inmemory local here.
            // This specifically uses a type with scalar representation to tempt Miri to use the
            // efficient way of storing local variables (outside adressable memory).
            Call(_unit = callee(Move(non_copy), Move(non_copy)), ReturnTo(after_call), UnwindContinue())
            //~[stack]^ ERROR: not granting access
            //~[tree]| ERROR: /read access .* forbidden/
        }
        after_call = {
            Return()
        }
    }
}

pub fn callee(x: S, mut y: S) {
    // With the setup above, if `x` and `y` are both moved,
    // then writing to `y` will change the value stored in `x`!
    y.0 = 0;
    assert_eq!(x.0, 42);
}
