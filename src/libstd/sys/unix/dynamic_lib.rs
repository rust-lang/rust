// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use string::String;

pub struct DynamicLibrary(*mut u8);

pub type Error = String;

pub use self::imp::open;

#[cfg(target_os = "nacl")]
pub mod imp {
    use super::DynamicLibrary;
    use string::String;

    pub fn open(_filename: Option<&OsStr>) -> Result<DynamicLibrary, String> {
        Err("NaCl + Newlib doesn't impl loading shared objects".into())
    }

    impl DynamicLibrary {
        pub fn symbol(&self, symbol: &str) -> Result<*mut u8, String> {
            Err(String::new())
        }

        pub fn close(&self) -> Result<(), String> {
            Err(String::new())
        }
    }
}

#[cfg(not(target_os = "nacl"))]
mod imp {
    use super::DynamicLibrary;
    use str;
    use ptr;
    use string::String;
    use borrow::ToOwned;
    use ffi::{CStr, CString, OsStr};
    use os::unix::ffi::OsStringExt;
    use libc;

    impl DynamicLibrary {
        pub fn symbol(&self, symbol: &str) -> Result<*mut u8, String> {
            let raw_string = CString::new(symbol).unwrap();
            check_for_errors_in(|| unsafe { dlsym(self.0 as *mut libc::c_void, raw_string.as_ptr()) } as *mut u8)
        }

        pub fn close(&self) -> Result<(), String> {
            check_for_errors_in(|| unsafe { dlclose(self.0 as *mut libc::c_void) }).map(drop)
        }
    }

    pub fn open(filename: Option<&OsStr>) -> Result<DynamicLibrary, String> {
        check_for_errors_in(|| {
            unsafe {
                match filename {
                    Some(filename) => open_external(filename),
                    None => open_internal(),
                }
            }
        }).map(DynamicLibrary)
    }

    fn check_for_errors_in<T, F>(f: F) -> Result<T, String> where
        F: FnOnce() -> T,
    {
        use sync::StaticMutex;
        static LOCK: StaticMutex = StaticMutex::new();
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
        let s = CString::new(filename.to_owned().into_vec()).unwrap();
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
}

#[cfg(target_os = "macos")]
pub const ENVVAR: &'static str = "DYLD_LIBRARY_PATH";
#[cfg(not(target_os = "macos"))]
pub const ENVVAR: &'static str = "LD_LIBRARY_PATH";

pub const SEPARATOR: &'static str = ":";
