//! The NAMED_ASM_LABELS late lint runs after name resolution, so it shouldn't
//! lint here when using `-Zgo-to-stage=nameres`

#![deny(named_asm_labels)]

use std::arch::asm;

fn foo() {
    unsafe {
        asm!("foo: nop");
    }
}
