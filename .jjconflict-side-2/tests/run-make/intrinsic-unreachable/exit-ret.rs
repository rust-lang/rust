#![crate_type = "lib"]
use std::arch::asm;

#[deny(unreachable_code)]
#[inline(never)]
pub fn exit(n: usize) -> i32 {
    unsafe {
        // Pretend this asm is an exit() syscall.
        asm!("/*{0}*/", in(reg) n);
    }
    // This return value is just here to generate some extra code for a return
    // value, making it easier for the test script to detect whether the
    // compiler deleted it.
    42
}
