//@ only-x86_64-unknown-linux-gnu
//@ needs-dynamic-linking
//@ run-pass
//@ compile-flags: -Cpanic=abort
//@ edition: 2024
//@ ignore-backends: gcc

#![allow(incomplete_features)]
#![feature(raw_dylib_elf)]
#![no_std]
#![no_main]

use core::ffi::{c_char, c_int};

extern "C" fn callback(
    _fpath: *const c_char,
    _sb: *const (),
    _tflag: c_int,
    _ftwbuf: *const (),
) -> c_int {
    0
}

// `libc.so` is a linker script that provides the paths to `libc.so.6` and `libc_nonshared.a`.
// In earlier versions of glibc, `libc_nonshared.a` provides the symbols `__libc_csu_init` and
// `__libc_csu_fini` required by `Scrt1.o`.
#[link(name = "c_nonshared", kind = "static")]
unsafe extern "C" {}

#[link(name = "libc.so.6", kind = "raw-dylib", modifiers = "+verbatim")]
unsafe extern "C" {
    #[link_name = "nftw@GLIBC_2.2.5"]
    unsafe fn nftw_2_2_5(
        dirpath: *const c_char,
        f: extern "C" fn(*const c_char, *const (), c_int, *const ()) -> c_int,
        nopenfd: c_int,
        flags: c_int,
    ) -> c_int;
    #[link_name = "nftw@GLIBC_2.3.3"]
    unsafe fn nftw_2_3_3(
        dirpath: *const c_char,
        f: extern "C" fn(*const c_char, *const (), c_int, *const ()) -> c_int,
        nopenfd: c_int,
        flags: c_int,
    ) -> c_int;
    #[link_name = "exit@GLIBC_2.2.5"]
    safe fn exit(status: i32) -> !;
    unsafe fn __libc_start_main() -> c_int;
}

#[unsafe(no_mangle)]
extern "C" fn main() -> ! {
    unsafe {
        // The old `nftw` does not check whether unknown flags are set.
        let res = nftw_2_2_5(c".".as_ptr(), callback, 20, 1 << 30);
        assert_eq!(res, 0);
    }
    unsafe {
        // The new `nftw` does.
        let res = nftw_2_3_3(c".".as_ptr(), callback, 20, 1 << 30);
        assert_eq!(res, -1);
    }
    exit(0);
}

#[cfg(not(test))]
#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo<'_>) -> ! {
    exit(1);
}

#[unsafe(no_mangle)]
extern "C" fn rust_eh_personality(
    _version: i32,
    _actions: i32,
    _exception_class: u64,
    _exception_object: *mut (),
    _context: *mut (),
) -> i32 {
    exit(1);
}
