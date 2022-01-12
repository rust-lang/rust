#![feature(core_intrinsics)]
#![crate_type="lib"]
use std::arch::asm;

use std::intrinsics;

#[allow(unreachable_code)]
pub fn exit(n: usize) -> i32 {
    unsafe {
        // Pretend this asm is an exit() syscall.
        asm!("/*{0}*/", in(reg) n);
        intrinsics::unreachable()
    }
    // This return value is just here to generate some extra code for a return
    // value, making it easier for the test script to detect whether the
    // compiler deleted it.
    42
}
