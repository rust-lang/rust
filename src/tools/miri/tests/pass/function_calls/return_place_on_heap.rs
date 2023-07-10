#![feature(raw_ref_op)]
#![feature(core_intrinsics)]
#![feature(custom_mir)]

use std::intrinsics::mir::*;

// Make sure calls with the return place "on the heap" work.
#[custom_mir(dialect = "runtime", phase = "optimized")]
pub fn main() {
    mir! {
        {
            let x = 0;
            let ptr = &raw mut x;
            Call(*ptr, after_call, myfun())
        }

        after_call = {
            Return()
        }
    }
}

fn myfun() -> i32 {
    13
}
