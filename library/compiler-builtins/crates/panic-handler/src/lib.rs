//! This is needed for tests on targets that require a `#[panic_handler]` function

#![feature(no_core)]
#![no_core]

extern crate core;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
