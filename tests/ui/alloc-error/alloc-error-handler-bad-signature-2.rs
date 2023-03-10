// compile-flags:-C panic=abort

#![feature(alloc_error_handler)]
#![no_std]
#![no_main]

struct Layout;

#[alloc_error_handler] //~ ERROR mismatched types
fn oom( //~ ERROR mismatched types
    info: Layout,
) {
    loop {}
}

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }
