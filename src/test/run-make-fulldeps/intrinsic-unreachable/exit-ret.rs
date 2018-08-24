#![feature(asm)]
#![crate_type="lib"]

#[deny(unreachable_code)]
pub fn exit(n: usize) -> i32 {
    unsafe {
        // Pretend this asm is an exit() syscall.
        asm!("" :: "r"(n) :: "volatile");
        // Can't actually reach this point, but rustc doesn't know that.
    }
    // This return value is just here to generate some extra code for a return
    // value, making it easier for the test script to detect whether the
    // compiler deleted it.
    42
}
