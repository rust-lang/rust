//! Check that dynamic libraries can link and run now that there is an unconditional use
//! of EII in `core`.

//@ build-pass
//@ compile-flags: -Cpanic=abort
//@ no-prefer-dynamic
//@ needs-crate-type: dylib
#![crate_type = "dylib"]
#![feature(core_io_internals)]
#![feature(core_io)]
#![feature(raw_os_error_ty)]
#![no_std]
#![no_implicit_prelude]

#[panic_handler]
fn panic_handler(_info: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

pub use core::io::raw_os_error::*;
