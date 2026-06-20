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
#![feature(lang_items)]
#![no_std]
#![no_implicit_prelude]

// See no_std/simple-runs.rs for details on why this is required.
#[cfg_attr(all(not(target_vendor = "apple"), unix), link(name = "c"))]
#[cfg_attr(target_vendor = "apple", link(name = "System"))]
extern "C" {}

#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

#[lang = "eh_personality"]
extern "C" fn rust_eh_personality(_: i32, _: i32, _: u64, _: *mut (), _: *mut ()) -> i32 {
    loop {}
}

pub use core::io::raw_os_error::*;
