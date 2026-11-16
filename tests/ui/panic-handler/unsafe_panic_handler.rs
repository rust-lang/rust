//@ check-pass
#![crate_type = "lib"]
#![no_std]

use core::panic::PanicInfo;

#[panic_handler]
unsafe fn handle(_: &PanicInfo) -> ! {
    //~^ WARN `#[panic_handler]` functions can't be `unsafe`
    //~| WARN this was previously accepted by the compiler but is being phased out
    loop {}
}
