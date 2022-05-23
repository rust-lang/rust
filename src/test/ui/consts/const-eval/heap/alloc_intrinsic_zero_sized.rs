// run-pass
#![feature(core_intrinsics)]
#![feature(const_heap)]
#![feature(inline_const)]

use std::intrinsics;

struct ZST;

fn main() {
    const {
        unsafe {
            let _ = intrinsics::const_allocate(0, 0) as *mut ZST;
        }
    }
}
