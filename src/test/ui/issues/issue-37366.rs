// ignore-emscripten
// compile-pass
// skip-codegen
#![feature(asm)]

macro_rules! interrupt_handler {
    () => {
        unsafe fn _interrupt_handler() {
            asm!("pop  eax" :::: "intel");
        }
    }
}
interrupt_handler!{}


fn main() {
}
