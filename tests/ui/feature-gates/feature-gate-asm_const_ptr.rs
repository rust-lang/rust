//@ only-x86_64

use std::arch::{asm, global_asm, naked_asm};

global_asm!("/* {} */", const &0);
//~^ ERROR using pointers in asm `const` operand is experimental

#[unsafe(naked)]
extern "C" fn naked() {
    unsafe {
        naked_asm!("ret /* {} */", const &0);
        //~^ ERROR using pointers in asm `const` operand is experimental
    }
}

fn main() {
    naked();
    unsafe {
        asm!("/* {} */", const &0);
        //~^ ERROR using pointers in asm `const` operand is experimental
    }
}
