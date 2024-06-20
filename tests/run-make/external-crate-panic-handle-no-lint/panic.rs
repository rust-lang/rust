#![crate_type = "lib"]
#![feature(panic_handler)]
#![no_std]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(_: &PanicInfo) -> ! {
    loop {}
}
