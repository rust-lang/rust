// compile-flags:-C panic=abort

#![feature(panic_implementation)]
#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_implementation]
fn panic(
    info: PanicInfo, //~ ERROR argument should be `&PanicInfo`
) -> () //~ ERROR return type should be `!`
{
}
