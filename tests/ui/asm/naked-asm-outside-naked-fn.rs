//@ edition: 2021
//@ needs-asm-support
//@ ignore-nvptx64
//@ ignore-spirv

#![feature(naked_functions)]
#![crate_type = "lib"]

use std::arch::naked_asm;

fn main() {
    test1();
}

#[naked]
extern "C" fn test1() {
    unsafe { naked_asm!("") }
}

extern "C" fn test2() {
    unsafe { naked_asm!("") }
    //~^ ERROR the `naked_asm!` macro can only be used in functions marked with `#[naked]`
}

extern "C" fn test3() {
    unsafe { (|| naked_asm!(""))() }
    //~^ ERROR the `naked_asm!` macro can only be used in functions marked with `#[naked]`
}

fn test4() {
    async move {
        unsafe {  naked_asm!("") } ;
        //~^ ERROR the `naked_asm!` macro can only be used in functions marked with `#[naked]`
    };
}
