//@ only-x86_64
//@ needs-asm-support

#![deny(unreachable_code)]
#![feature(asm_goto)]

use std::arch::asm;

fn goto_fallthough() {
    unsafe {
        asm!(
            "/* {} */",
            label {
                core::hint::unreachable_unchecked();
                //~^ ERROR [E0133]
            }
        )
    }
}

fn main() {
    goto_fallthough();
}
