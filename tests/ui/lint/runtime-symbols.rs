// This test checks the runtime symbols lint.

//@ edition: 2021
//@ normalize-stderr: "\*const [iu]8" -> "*const U8"

#![feature(c_variadic)]
#![allow(clashing_extern_declarations)] // we are voluntarily testing different definitions

use core::ffi::{c_char, c_int, c_void};

fn invalid() {
    #[no_mangle]
    pub fn memmove() {}
    //~^ ERROR invalid definition of the runtime `memmove` symbol

    extern "C" {
        pub fn memset();
        //~^ ERROR invalid definition of the runtime `memset` symbol

        pub fn memcmp();
        //~^ ERROR invalid definition of the runtime `memcmp` symbol
    }

    #[no_mangle]
    pub static strlen: () = ();
    //~^ ERROR invalid definition of the runtime `strlen` symbol

    // ABI mismatch: Rust ABI instead of C ABI
    #[no_mangle]
    pub fn memcpy(dest: *mut c_void, src: *const c_void, n: usize) -> *mut c_void {
        dest
    }
    //~^^^ ERROR invalid definition of the runtime `memcpy` symbol

    // C-Variadic mismatch
    #[no_mangle]
    pub unsafe extern "C" fn bcmp(s1: *const c_void, s2: *const c_void, n: usize, _: ...) -> c_int {
        0
    }
    //~^^^ ERROR invalid definition of the runtime `bcmp` symbol

    // Return type is missing
    #[export_name = "bcmp"]
    pub extern "C" fn bcmp_(s1: *const c_void, s2: *const c_void, n: usize) {}
    //~^ ERROR invalid definition of the runtime `bcmp` symbol
}

fn suspicious() {
    #[no_mangle]
    pub extern "C" fn memcpy(dest: *mut c_void, src: *const c_void, n: i64) -> *mut c_void {
        std::ptr::null_mut()
    }
    //~^^^ WARN suspicious definition of the runtime `memcpy` symbol

    #[no_mangle]
    pub extern "C" fn memmove(dest: *mut c_void, src: *const c_void, n: i64) -> *mut c_void {
        std::ptr::null_mut()
    }
    //~^^^ WARN suspicious definition of the runtime `memmove` symbol

    extern "C" {
        fn memset(s: *mut c_void, c: c_int, n: usize) -> f64;
        //~^ WARN suspicious definition of the runtime `memset` symbol
    }

    #[export_name = "bcmp"]
    pub extern "C" fn bcmp_(s1: *const u8, s2: *const u8, n: usize) -> c_int {
        0
    }
    //~^^^ WARN suspicious definition of the runtime `bcmp` symbol

    #[no_mangle]
    pub extern "C" fn strlen(s: *const u64) -> usize {
        0
    }
    //~^^^ WARN suspicious definition of the runtime `strlen` symbol
}

fn valid() {
    extern "C" {
        fn memcpy(dest: *mut c_void, src: *const c_void, n: usize) -> *mut c_void;

        fn memmove(dest: *mut c_void, src: *const c_void, n: usize) -> *mut c_void;

        fn memset(s: *mut c_void, c: c_int, n: usize) -> *mut c_void;

        fn memcmp(s1: *const c_void, s2: *const c_void, n: usize) -> c_int;

        fn bcmp(s1: *const c_void, s2: *const c_void, n: usize) -> c_int;

        static strlen: Option<unsafe extern "C" fn(s: *const c_char) -> usize>;
    }
}

fn main() {}
