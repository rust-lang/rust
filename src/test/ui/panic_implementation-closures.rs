// compile-pass

#![crate_type = "rlib"]
#![no_std]

#[panic_handler]
pub fn panic_fmt(_: &::core::panic::PanicInfo) -> ! {
    |x: u8| x;
    loop {}
}
