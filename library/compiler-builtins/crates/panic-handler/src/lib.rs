// Hack of a crate until rust-lang/rust#51647 is fixed

#![feature(no_core)]
#![no_core]

extern crate core;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
