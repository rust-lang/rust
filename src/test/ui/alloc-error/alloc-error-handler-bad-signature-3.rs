// compile-flags:-C panic=abort

#![feature(alloc_error_handler, panic_implementation)]
#![no_std]
#![no_main]

struct Layout;

#[alloc_error_handler]
fn oom() -> ! { //~ ERROR function should have one argument
    loop {}
}

#[panic_implementation]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }
