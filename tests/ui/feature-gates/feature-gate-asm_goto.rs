//@ only-x86_64

use std::arch::asm;

fn main() {
    unsafe {
        asm!("jmp {}", label {});
        //~^ ERROR label operands for inline assembly are unstable
    }
}
