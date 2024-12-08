//@ compile-flags:-C panic=abort

#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(info: PanicInfo) -> () {}
//~^ `#[panic_handler]` function has wrong type [E0308]
