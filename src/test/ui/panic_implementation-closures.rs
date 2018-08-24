// compile-pass

#![crate_type = "rlib"]
#![no_std]
#![feature(panic_implementation)]

#[panic_implementation]
pub fn panic_fmt(_: &::core::panic::PanicInfo) -> ! {
    |x: u8| x;
    loop {}
}
