// run-pass
#![feature(core_intrinsics)]
#![feature(const_heap)]
#![feature(inline_const)]

use std::intrinsics;

fn main() {
    const {
        unsafe {
            let ptr1 = intrinsics::const_allocate(0, 0);
            let ptr2 = intrinsics::const_allocate(0, 0);
            intrinsics::const_deallocate(ptr1, 0, 0);
            intrinsics::const_deallocate(ptr2, 0, 0);
        }
    }
}
