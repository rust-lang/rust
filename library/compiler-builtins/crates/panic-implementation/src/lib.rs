// Hack of a crate until rust-lang/rust#51647 is fixed

#![feature(no_core, panic_implementation)]
#![no_core]

extern crate core;

#[panic_implementation]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
