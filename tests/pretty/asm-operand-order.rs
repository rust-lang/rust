//@ pretty-mode:expanded
//@ pp-exact:asm-operand-order.pp
//@ only-x86_64

use std::arch::asm;

pub fn main() {
    unsafe {
        asm!("{val}", out("rax") _, val = in(reg) 4);
        asm!("{val}", out("rax") _, val = in(reg) 4, options(nostack));
        asm!("{a} {b}", out("rax") _, a = in(reg) 4, b = in(reg) 5);
    }
}
