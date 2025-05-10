//@ compile-flags:-C panic=abort

#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(info: PanicInfo) -> () {}
//~^ ERROR function `panic` has a type that is incompatible with the declaration of `#[panic_handler]`
