//! Check that `no_std` binaries can link and run without depending on `libstd`.

//@ run-pass
//@ compile-flags: -Cpanic=abort
//@ ignore-wasm different `main` convention

#![feature(lang_items)]
#![no_std]
#![no_main]

use core::ffi::{c_char, c_int};
use core::panic::PanicInfo;

// # Linux
//
// Linking `libc` is required by crt1.o, otherwise the linker fails with:
// > /usr/bin/ld: in function `_start': undefined reference to `__libc_start_main'
//
// # Apple
//
// Linking `libSystem` is required, otherwise the linker fails with:
// > ld: dynamic executables or dylibs must link with libSystem.dylib
//
// With the new linker introduced in Xcode 15, the error is instead:
// > Undefined symbols: "dyld_stub_binder", referenced from: <initial-undefines>
//
// This _can_ be worked around by raising the deployment target with
// MACOSX_DEPLOYMENT_TARGET=13.0, though it's a bit hard to test that while
// still allowing the test suite to support running with older Xcode versions.
#[cfg_attr(all(not(target_vendor = "apple"), unix), link(name = "c"))]
#[cfg_attr(target_vendor = "apple", link(name = "System"))]
extern "C" {}

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
