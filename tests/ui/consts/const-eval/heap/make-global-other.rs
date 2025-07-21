// Ensure that we can't call `const_make_global` on pointers not in the current interpreter.

#![feature(core_intrinsics)]
#![feature(const_heap)]

use std::intrinsics;

const X: &i32 = &0;

const Y: &i32 = unsafe {
    &*(intrinsics::const_make_global(X as *const i32 as *mut u8) as *const i32)
    //~^ error: pointer passed to `const_make_global` does not point to a heap allocation: ALLOC0<imm>
};

fn main() {}
