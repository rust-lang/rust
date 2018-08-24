// compile-flags:-C panic=abort

#![feature(panic_implementation)]
#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_implementation]
fn panic<T>(pi: &PanicInfo) -> ! {
    //~^ ERROR `#[panic_implementation]` function should have no type parameters
    loop {}
}
