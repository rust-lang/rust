//@ compile-flags:-C panic=abort

#![no_std]
#![no_main]

#[panic_handler] //~ ERROR `#[panic_handler]` must be used on a function
static X: u32 = 42;
