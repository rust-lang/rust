// This does need an aliasing model and protectors.
//@revisions: stack tree
//@[tree]compile-flags: -Zmiri-tree-borrows
#![feature(core_intrinsics)]
#![feature(custom_mir)]

use std::intrinsics::mir::*;

#[custom_mir(dialect = "runtime", phase = "optimized")]
pub fn main() {
    mir! {
        {
            let _x = 0;
            let ptr = &raw mut _x;
            // We arrange for `myfun` to have a pointer that aliases
            // its return place. Writing to that pointer is UB.
            Call(_x = myfun(ptr), ReturnTo(after_call), UnwindContinue())
        }

        after_call = {
            Return()
        }
    }
}

fn myfun(ptr: *mut i32) -> i32 {
    // This overwrites the return place, which shouldn't be possible through another pointer.
    unsafe { ptr.write(0) };
    //~[stack]^ ERROR: does not exist in the borrow stack
    //~[tree]| ERROR: /write access .* forbidden/
    13
}
