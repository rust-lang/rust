// Checks that the #[unsafe(naked)] attribute can be placed on function definitions only.
//
//@ needs-asm-support
#![unsafe(naked)] //~ ERROR should be applied to a function definition

use std::arch::naked_asm;

extern "C" {
    #[unsafe(naked)] //~ ERROR should be applied to a function definition
    fn f();
}

#[unsafe(naked)] //~ ERROR should be applied to a function definition
#[repr(C)]
struct S {
    #[unsafe(naked)] //~ ERROR should be applied to a function definition
    a: u32,
    b: u32,
}

trait Invoke {
    #[unsafe(naked)] //~ ERROR should be applied to a function definition
    extern "C" fn invoke(&self);
}

impl Invoke for S {
    #[unsafe(naked)]
    extern "C" fn invoke(&self) {
        naked_asm!("")
    }
}

#[unsafe(naked)]
extern "C" fn ok() {
    naked_asm!("")
}

impl S {
    #[unsafe(naked)]
    extern "C" fn g() {
        naked_asm!("")
    }

    #[unsafe(naked)]
    extern "C" fn h(&self) {
        naked_asm!("")
    }
}

fn main() {
    #[unsafe(naked)] //~ ERROR should be applied to a function definition
    || {};
}
