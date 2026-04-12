// This test checks the runtime symbols lint.

//@ edition: 2021
//@ normalize-stderr: "\*const [iu]8" -> "*const U8"

#![allow(clashing_extern_declarations)] // we are volontary testing differents defs

use core::ffi::{c_char, c_int, c_void};

fn invalid() {
    #[no_mangle]
    pub extern "C" fn memcpy(dest: *mut c_void, src: *const c_void, n: i64) -> *mut c_void {
        std::ptr::null_mut()
    }
    //~^^^ ERROR invalid definition of the runtime `memcpy` symbol

    #[no_mangle]
    pub fn memmove() {}
    //~^ ERROR invalid definition of the runtime `memmove` symbol

    extern "C" {
        pub fn memset();
        //~^ ERROR invalid definition of the runtime `memset` symbol

        pub fn memcmp();
        //~^ ERROR invalid definition of the runtime `memcmp` symbol
    }

    #[export_name = "bcmp"]
    pub fn bcmp_() {}
    //~^ ERROR invalid definition of the runtime `bcmp` symbol

    #[no_mangle]
    pub static strlen: () = ();
    //~^ ERROR invalid definition of the runtime `strlen` symbol
}

fn valid() {
    extern "C" {
        fn memcpy(dest: *mut c_void, src: *const c_void, n: usize) -> *mut c_void;

        fn memmove(dest: *mut c_void, src: *const c_void, n: usize) -> *mut c_void;

        fn memset(s: *mut c_void, c: c_int, n: usize) -> *mut c_void;

        fn memcmp(s1: *const c_void, s2: *const c_void, n: usize) -> c_int;

        fn bcmp(s1: *const c_void, s2: *const c_void, n: usize) -> c_int;

        fn strlen(s: *const c_char) -> usize;
    }
}

fn main() {}
