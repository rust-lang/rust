//@ needs-asm-support
//@ only-x86_64

// tests that labels containing only the digits 0 and 1 are rejected
// uses of such labels can sometimes be interpreted as a binary literal

use std::arch::{asm, global_asm};

fn main() {
    unsafe {
        asm!("0: jmp 0b"); //~ ERROR avoid using labels containing only the digits
        asm!("1: jmp 1b"); //~ ERROR avoid using labels containing only the digits
        asm!("10: jmp 10b"); //~ ERROR avoid using labels containing only the digits
        asm!("01: jmp 01b"); //~ ERROR avoid using labels containing only the digits
        asm!("1001101: jmp 1001101b"); //~ ERROR avoid using labels containing only the digits
    }
}
