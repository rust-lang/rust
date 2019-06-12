// check-pass
// ignore-emscripten

#![feature(asm)]

macro_rules! interrupt_handler {
    () => {
        unsafe fn _interrupt_handler() {
            asm!("pop  eax" :::: "intel");
        }
    }
}
interrupt_handler!{}

fn main() {}
