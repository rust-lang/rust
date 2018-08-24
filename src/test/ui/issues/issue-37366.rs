// ignore-emscripten

#![feature(rustc_attrs, asm)]

macro_rules! interrupt_handler {
    () => {
        unsafe fn _interrupt_handler() {
            asm!("pop  eax" :::: "intel");
        }
    }
}
interrupt_handler!{}

#[rustc_error]
fn main() { //~ ERROR compilation successful
}
