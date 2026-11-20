//@ compile-flags:-C panic=abort
//@ only-x86_64

#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_handler]
#[track_caller]
//~^ ERROR `#[panic_handler]` function is not allowed to have `#[track_caller]`
fn panic(info: &PanicInfo) -> ! {
    unimplemented!();
}
