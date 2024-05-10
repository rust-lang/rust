#![allow(internal_features)]
#![feature(core_intrinsics, custom_mir)]

use std::intrinsics::mir::*;
use std::num::NonZero;
use std::ptr;

// This function supposedly returns a `NonZero<u32>`, but actually returns something invalid in a way that
// never materializes a bad `NonZero<u32>` value: we take a pointer to the return place and cast the pointer
// type. That way we never get an "invalid value constructed" error inside the function, it can
// only possibly be detected when the return value is passed to the caller.
#[custom_mir(dialect = "runtime", phase = "optimized")]
fn f() -> NonZero<u32> {
    mir! {
        {
            let tmp = ptr::addr_of_mut!(RET);
            let ptr = tmp as *mut u32;
            *ptr = 0;
            Return()
        }
    }
}

fn main() {
    let f: fn() -> u32 = unsafe { std::mem::transmute(f as fn() -> NonZero<u32>) };
    // There's a `NonZero<u32>` to `u32` transmute happening here.
    f(); //~ERROR: expected something greater or equal to 1
}
