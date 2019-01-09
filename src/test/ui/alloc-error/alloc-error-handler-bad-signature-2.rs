// compile-flags:-C panic=abort

#![feature(alloc_error_handler, panic_handler)]
#![no_std]
#![no_main]

struct Layout;

#[alloc_error_handler]
fn oom(
    info: Layout, //~ ERROR argument should be `Layout`
) { //~ ERROR return type should be `!`
    loop {}
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }
