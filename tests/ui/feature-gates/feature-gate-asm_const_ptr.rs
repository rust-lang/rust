//@ only-x86_64

use std::arch::asm;

fn main() {
    unsafe {
        asm!("/* {} */", const &0);
        //~^ ERROR using pointers in asm `const` operand is experimental
    }
}
