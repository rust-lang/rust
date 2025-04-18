//@ compile-flags:-C panic=abort

#![no_std]
#![no_main]

#[panic_handler] //~ ERROR `#[panic_handler]` is only valid on function
#[no_mangle]
static X: u32 = 42;

//~? ERROR `#[panic_handler]` required, but not found
