//@ needs-unwind
//@ compile-flags: -Cpanic=unwind

// Make sure that we don't emit an error message mentioning internal lang items.

#![no_std]
#![no_main]

#[panic_handler]
fn handler(_info: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

fn main() {}

//~? ERROR unwinding panics are not supported without std
