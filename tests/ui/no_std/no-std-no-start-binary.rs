//@ compile-flags: -Cpanic=abort --emit link
//@ error-pattern:using `fn main` requires the standard library

// Make sure that we don't emit an error message mentioning internal lang items.

#![no_std]

#[panic_handler]
fn handler(_info: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

fn main() {}

//~? ERROR using `fn main` requires the standard library
