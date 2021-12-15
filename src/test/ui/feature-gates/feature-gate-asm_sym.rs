// only-x86_64

use std::arch::asm;

fn main() {
    unsafe {
        asm!("mov eax, {}", sym main);
        //~^ ERROR sym operands for inline assembly are unstable
    }
}
