//@ compile-flags:-C panic=abort

#![feature(alloc_error_handler)]
#![no_std]
#![no_main]

use core::alloc::Layout;

#[alloc_error_handler]
fn oom(
    info: &Layout, //~^ ERROR mismatched types
) -> () //~^^ ERROR mismatched types
{
    loop {}
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }
