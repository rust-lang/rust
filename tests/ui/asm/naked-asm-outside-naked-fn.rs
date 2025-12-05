//@ edition: 2021
//@ needs-asm-support
//@ ignore-nvptx64
//@ ignore-spirv

#![crate_type = "lib"]

use std::arch::naked_asm;

fn main() {
    test1();
}

#[unsafe(naked)]
extern "C" fn test1() {
    naked_asm!("")
}

extern "C" fn test2() {
    naked_asm!("")
    //~^ ERROR the `naked_asm!` macro can only be used in functions marked with `#[unsafe(naked)]`
}

extern "C" fn test3() {
    (|| naked_asm!(""))()
    //~^ ERROR the `naked_asm!` macro can only be used in functions marked with `#[unsafe(naked)]`
}

fn test4() {
    async move {
        naked_asm!("");
        //~^ ERROR the `naked_asm!` macro can only be used in functions marked with `#[unsafe(naked)]`
    };
}
