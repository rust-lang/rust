//! Check that closures can be used inside `#[panic_handler]` functions.

//@ check-pass

#![crate_type = "rlib"]
#![no_std]

#[panic_handler]
pub fn panicfmt(_: &::core::panic::PanicInfo) -> ! {
    |x: u8| x;
    loop {}
}
