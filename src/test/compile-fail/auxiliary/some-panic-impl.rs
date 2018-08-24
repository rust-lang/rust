// no-prefer-dynamic

#![crate_type = "rlib"]
#![feature(panic_implementation)]
#![no_std]

use core::panic::PanicInfo;

#[panic_implementation]
fn panic(info: &PanicInfo) -> ! {
    loop {}
}
