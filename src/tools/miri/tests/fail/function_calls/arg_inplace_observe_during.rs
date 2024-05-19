//@revisions: stack tree none
//@[tree]compile-flags: -Zmiri-tree-borrows
//@[none]compile-flags: -Zmiri-disable-stacked-borrows
#![feature(custom_mir, core_intrinsics)]
use std::intrinsics::mir::*;

pub struct S(i32);

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn main() {
    mir! {
        let _unit: ();
        {
            let non_copy = S(42);
            let ptr = std::ptr::addr_of_mut!(non_copy);
            // This could change `non_copy` in-place
            Call(_unit = change_arg(Move(*ptr), ptr), ReturnTo(after_call), UnwindContinue())
        }
        after_call = {
            Return()
        }

    }
}

pub fn change_arg(mut x: S, ptr: *mut S) {
    x.0 = 0;
    // If `x` got passed in-place, we'd see the write through `ptr`!
    // Make sure we are not allowed to do that read.
    unsafe { ptr.read() };
    //~[stack]^ ERROR: not granting access
    //~[tree]| ERROR: /read access .* forbidden/
    //~[none]| ERROR: uninitialized
}
