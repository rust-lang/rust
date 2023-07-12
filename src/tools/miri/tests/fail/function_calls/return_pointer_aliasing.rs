//@revisions: stack tree none
//@[tree]compile-flags: -Zmiri-tree-borrows
//@[none]compile-flags: -Zmiri-disable-stacked-borrows
#![feature(raw_ref_op)]
#![feature(core_intrinsics)]
#![feature(custom_mir)]

use std::intrinsics::mir::*;

#[custom_mir(dialect = "runtime", phase = "optimized")]
pub fn main() {
    mir! {
        {
            let x = 0;
            let ptr = &raw mut x;
            // We arrange for `myfun` to have a pointer that aliases
            // its return place. Even just reading from that pointer is UB.
            Call(*ptr, after_call, myfun(ptr))
        }

        after_call = {
            Return()
        }
    }
}

fn myfun(ptr: *mut i32) -> i32 {
    unsafe { ptr.read() };
    //~[stack]^ ERROR: not granting access
    //~[tree]| ERROR: /read access .* forbidden/
    //~[none]| ERROR: uninitialized
    // Without an aliasing model, reads are "fine" but at least they return uninit data.
    13
}
