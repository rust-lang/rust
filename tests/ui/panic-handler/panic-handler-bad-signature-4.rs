//@ compile-flags:-C panic=abort

#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_handler]
fn panic<T>(pi: &PanicInfo) -> ! {
    //~^ ERROR #[panic_handler] cannot have generic parameters other than lifetimes
    loop {}
}
