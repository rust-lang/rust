//@ compile-flags:-C panic=abort

#![no_std]
#![no_main]

#[panic_handler] //~ ERROR attribute cannot be used on statics
static X: u32 = 42;

//~? ERROR `#[panic_handler]` function required, but not found
