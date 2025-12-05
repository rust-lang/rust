#![crate_type = "rlib"]
#![feature(alloc_error_handler)]
#![no_std]

#[alloc_error_handler]
pub fn error_handler(_: core::alloc::Layout) -> ! {
    panic!();
}
