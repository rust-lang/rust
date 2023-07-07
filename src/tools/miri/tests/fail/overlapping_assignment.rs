#![feature(core_intrinsics)]
#![feature(custom_mir)]

use std::intrinsics::mir::*;

// It's not that easy to fool the MIR validity check
// which wants to prevent overlapping assignments...
// So we use two separate pointer arguments, and then arrange for them to alias.
#[custom_mir(dialect = "runtime", phase = "optimized")]
pub fn self_copy(ptr1: *mut [i32; 4], ptr2: *mut [i32; 4]) {
    mir! {
        {
            *ptr1 = *ptr2; //~ERROR: overlapping ranges
            Return()
        }
    }
}

pub fn main() {
    let mut x = [0; 4];
    let ptr = std::ptr::addr_of_mut!(x);
    self_copy(ptr, ptr);
}
