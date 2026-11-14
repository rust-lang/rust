#![crate_type = "lib"]
#![no_std]

use core::panic::PanicInfo;

#[panic_handler]
unsafe fn handle(_: &PanicInfo) -> ! {
    //~^ ERROR function `handle` has a type that is incompatible with the declaration of `#[panic_handler]`
    loop {}
}
