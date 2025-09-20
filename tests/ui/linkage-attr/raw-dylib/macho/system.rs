//! Check that we can link and run a `no_std` binary without.

//@ only-apple
//@ run-pass
//@ compile-flags: -Cpanic=abort
//@ compile-flags: -Clinker=ld -Clink-arg=-syslibroot -Clink-arg=./Empty.sdk

#![allow(incomplete_features)]
#![feature(raw_dylib_macho)]
#![feature(lang_items)]
#![no_std]
#![no_main]

use core::ffi::{c_char, c_int};
use core::panic::PanicInfo;

#[link(
    name = "/usr/lib/libSystem.B.dylib",
    kind = "raw-dylib",
    modifiers = "+verbatim",
    // current_version: 1351,
    // compatibility_version: 1,
)]
#[allow(unused)]
unsafe extern "C" {
    #[link_name = "\x01dyld_stub_binder"]
    unsafe fn dyld_stub_binder();
}

#[panic_handler]
fn panic_handler(_info: &PanicInfo<'_>) -> ! {
    loop {}
}

#[lang = "eh_personality"]
extern "C" fn rust_eh_personality(
    _version: i32,
    _actions: i32,
    _exception_class: u64,
    _exception_object: *mut (),
    _context: *mut (),
) -> i32 {
    loop {}
}

#[no_mangle]
extern "C" fn main(_argc: c_int, _argv: *const *const c_char) -> c_int {
    0
}
