// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::v1::*;

use ffi::{OsStr, CString};
use libc;
use libc::consts::os::extra::ERROR_CALL_NOT_IMPLEMENTED;
use sys::windows::os;
use sys::dynamic_lib as sys;
use os::windows::prelude::*;
use ptr;
use sys::c::SetThreadErrorMode;

pub struct Dl(*mut u8);

impl sys::Dl for Dl {
    fn open(filename: Option<&OsStr>) -> Result<Self, String> {
        // disable "dll load failed" error dialog.
        let mut use_thread_mode = true;
        let prev_error_mode = unsafe {
            // SEM_FAILCRITICALERRORS 0x01
            let new_error_mode = 1;
            let mut prev_error_mode = 0;
            // Windows >= 7 supports thread error mode.
            let result = SetThreadErrorMode(new_error_mode, &mut prev_error_mode);
            if result == 0 {
                let err = os::errno();
                if err as libc::c_int == ERROR_CALL_NOT_IMPLEMENTED {
                    use_thread_mode = false;
                    // SetThreadErrorMode not found. use fallback solution:
                    // SetErrorMode() Note that SetErrorMode is process-wide so
                    // this can cause race condition!  However, since even
                    // Windows APIs do not care of such problem (#20650), we
                    // just assume SetErrorMode race is not a great deal.
                    prev_error_mode = SetErrorMode(new_error_mode);
                }
            }
            prev_error_mode
        };

        unsafe {
            SetLastError(0);
        }

        let result = match filename {
            Some(filename) => {
                let filename_str: Vec<_> =
                    filename.encode_wide().chain(Some(0)).collect();
                let result = unsafe {
                    LoadLibraryW(filename_str.as_ptr() as *const libc::c_void)
                };
                // beware: Vec/String may change errno during drop!
                // so we get error here.
                if result == ptr::null_mut() {
                    let errno = os::errno();
                    Err(os::error_string(errno))
                } else {
                    Ok(result as *mut u8)
                }
            }
            None => {
                let mut handle = ptr::null_mut();
                let succeeded = unsafe {
                    GetModuleHandleExW(0 as libc::DWORD, ptr::null(), &mut handle)
                };
                if succeeded == libc::FALSE {
                    let errno = os::errno();
                    Err(os::error_string(errno))
                } else {
                    Ok(handle as *mut u8)
                }
            }
        }.map(Dl);

        unsafe {
            if use_thread_mode {
                SetThreadErrorMode(prev_error_mode, ptr::null_mut());
            } else {
                SetErrorMode(prev_error_mode);
            }
        }

        result
    }

    fn symbol(&self, symbol: &str) -> Result<*mut u8, String> {
        let raw_string = CString::new(symbol).unwrap();
        check_for_errors_in(|| GetProcAddress(self.0 as *mut libc::c_void, raw_string.as_ptr()) as *mut u8)
    }
    fn close(&self) -> Result<(), String> {
        check_for_errors_in(|| FreeLibrary(self.0 as *mut libc::c_void))
    }

    fn envvar() -> &'static str {
        "PATH"
    }

    fn separator() -> &'static str {
        ";"
    }
}

fn check_for_errors_in<T, F>(f: F) -> Result<T, String> where
    F: FnOnce() -> T,
{
    unsafe {
        SetLastError(0);

        let result = f();

        let error = os::errno();
        if 0 == error {
            Ok(result)
        } else {
            Err(format!("Error code {}", error))
        }
    }
}

#[allow(non_snake_case)]
extern "system" {
    fn SetLastError(error: libc::size_t);
    fn LoadLibraryW(name: *const libc::c_void) -> *mut libc::c_void;
    fn GetModuleHandleExW(dwFlags: libc::DWORD, name: *const u16,
                          handle: *mut *mut libc::c_void) -> libc::BOOL;
    fn GetProcAddress(handle: *mut libc::c_void,
                      name: *const libc::c_char) -> *mut libc::c_void;
    fn FreeLibrary(handle: *mut libc::c_void);
    fn SetErrorMode(uMode: libc::c_uint) -> libc::c_uint;
}
}
