//@ compile-flags:-C panic=abort

#![no_std]
#![no_main]

use core::panic::PanicInfo;

#[panic_handler]
fn panic(info: &'static PanicInfo) -> !
//~^ ERROR mismatched types [E0308]
//~^^ ERROR cannot infer an appropriate lifetime for lifetime parameter '_
{
    loop {}
}
