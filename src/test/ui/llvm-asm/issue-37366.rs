// check-pass
// ignore-emscripten

#![feature(llvm_asm)]
#![allow(deprecated)] // llvm_asm!

macro_rules! interrupt_handler {
    () => {
        unsafe fn _interrupt_handler() {
            llvm_asm!("pop  eax" :::: "intel");
        }
    }
}
interrupt_handler!{}

fn main() {}
