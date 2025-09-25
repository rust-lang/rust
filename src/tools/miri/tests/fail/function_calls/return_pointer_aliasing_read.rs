//@revisions: stack tree none
//@[tree]compile-flags: -Zmiri-tree-borrows
//@[none]compile-flags: -Zmiri-disable-stacked-borrows
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
            // its return place. Even just reading from that pointer is UB.
            Call(_x = myfun(ptr), ReturnTo(after_call), UnwindContinue())
        }

        after_call = {
            Return()
        }
    }
}

fn myfun(ptr: *mut i32) -> i32 {
    unsafe { ptr.read() };
    //~[stack]^ ERROR: does not exist in the borrow stack
    //~[tree]| ERROR: /read access .* forbidden/
    //~[none]| ERROR: uninitialized
    // Without an aliasing model, reads are "fine" but at least they return uninit data.
    13
}
