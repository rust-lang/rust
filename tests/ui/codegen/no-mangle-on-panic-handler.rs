// Issue an error when the user uses #[no_mangle] on the panic handler
//@ edition:2024

#![crate_type="lib"]
#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[unsafe(no_mangle)] //~ ERROR `#[no_mangle]` cannot be used on internal language items
#[panic_handler]
pub unsafe fn panic_fmt(pi: &PanicInfo) -> ! {
    loop {}
}
