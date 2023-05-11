// Checks that #[naked] attribute can be placed on function definitions only.
//
// needs-asm-support
#![feature(naked_functions)]
#![naked] //~ ERROR should be applied to a function definition

use std::arch::asm;

extern "C" {
    #[naked] //~ ERROR should be applied to a function definition
    fn f();
}

#[naked] //~ ERROR should be applied to a function definition
#[repr(C)]
struct S {
    a: u32,
    b: u32,
}

trait Invoke {
    #[naked] //~ ERROR should be applied to a function definition
    extern "C" fn invoke(&self);
}

impl Invoke for S {
    #[naked]
    extern "C" fn invoke(&self) {
        unsafe { asm!("", options(noreturn)) }
    }
}

#[naked]
extern "C" fn ok() {
    unsafe { asm!("", options(noreturn)) }
}

impl S {
    #[naked]
    extern "C" fn g() {
        unsafe { asm!("", options(noreturn)) }
    }

    #[naked]
    extern "C" fn h(&self) {
        unsafe { asm!("", options(noreturn)) }
    }
}

fn main() {
    #[naked] || {}; //~ ERROR should be applied to a function definition
}
