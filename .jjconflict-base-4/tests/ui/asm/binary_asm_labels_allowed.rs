//@ build-pass
//@ only-aarch64

// The `binary_asm_labels` lint should only be raised on `x86`. Make sure it
// doesn't get raised on other platforms.

use std::arch::asm;

fn main() {
    unsafe {
        asm!("0: bl 0b");
        asm!("1: bl 1b");
        asm!("10: bl 10b");
        asm!("01: bl 01b");
        asm!("1001101: bl 1001101b");
    }
}
