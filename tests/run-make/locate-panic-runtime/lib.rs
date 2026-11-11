#![feature(no_core)]
#![no_std]
#![no_core]
#![crate_type = "dylib"]

extern crate std;

#[panic_handler]
fn panic(_: &std::PanicInfo) -> ! {
    loop {}
}
