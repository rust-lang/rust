// only-x86_64

use std::arch::asm;

fn main() {
    unsafe {
        asm!("mov eax, {}", const 123);
        //~^ ERROR const operands for inline assembly are unstable
    }
}
