//@ compile-flags:-C panic=abort

#![no_std]
#![deny(todo_macro_uses)]

#[panic_handler]
fn panic_handler(_info: &core::panic::PanicInfo) -> ! {
    loop {}
}

pub fn main() {
    todo!();
    //~^todo_macro_uses
}
