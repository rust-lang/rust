// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use dynamic_lib as sys;
use os_str::prelude::*;
use core::str;
use core::ptr;
use collections::String;
use collections::borrow::ToOwned;
use c_str::{CStr, CString};
use libc;

pub struct DynamicLibrary(*mut u8);

impl sys::DynamicLibrary for DynamicLibrary {
    type Error = String;

    fn open(filename: Option<&OsStr>) -> Result<Self, String> {
        check_for_errors_in(|| {
            unsafe {
                match filename {
                    Some(filename) => open_external(filename),
                    None => open_internal(),
                }
            }
        }).map(DynamicLibrary)
    }

    fn symbol(&self, symbol: &str) -> Result<*mut u8, String> {
        let raw_string = CString::new(symbol).unwrap();
        check_for_errors_in(|| unsafe { dlsym(self.0 as *mut libc::c_void, raw_string.as_ptr()) } as *mut u8)
    }

    fn close(&self) -> Result<(), String> {
        check_for_errors_in(|| unsafe { dlclose(self.0 as *mut libc::c_void) }).map(drop)
    }

    #[cfg(target_os = "macos")]
    fn envvar() -> &'static str { "DYLD_LIBRARY_PATH" }
    #[cfg(not(target_os = "macos"))]
    fn envvar() -> &'static str { "LD_LIBRARY_PATH" }

    fn separator() -> &'static str { ":" }
}

fn check_for_errors_in<T, F>(f: F) -> Result<T, String> where
    F: FnOnce() -> T,
{
    use sync::prelude::*;
    static LOCK: Mutex = Mutex::new();
    unsafe {
        // dlerror isn't thread safe, so we need to lock around this entire
        // sequence
        let _guard = LOCK.lock();
        let _old_error = dlerror();

        let result = f();

        let last_error = dlerror() as *const _;
        let ret = if ptr::null() == last_error {
            Ok(result)
        } else {
            let s = CStr::from_ptr(last_error).to_bytes();
            Err(str::from_utf8(s).unwrap().to_owned())
        };

        ret
    }
}

const LAZY: libc::c_int = 1;

unsafe fn open_external(filename: &OsStr) -> *mut u8 {
    let s = CString::new(::os_str::OsStr::to_bytes(filename).unwrap()).unwrap();
    dlopen(s.as_ptr(), LAZY) as *mut u8
}

unsafe fn open_internal() -> *mut u8 {
    dlopen(ptr::null(), LAZY) as *mut u8
}

extern {
    fn dlopen(filename: *const libc::c_char,
              flag: libc::c_int) -> *mut libc::c_void;
    fn dlerror() -> *mut libc::c_char;
    fn dlsym(handle: *mut libc::c_void,
             symbol: *const libc::c_char) -> *mut libc::c_void;
    fn dlclose(handle: *mut libc::c_void) -> libc::c_int;
}
