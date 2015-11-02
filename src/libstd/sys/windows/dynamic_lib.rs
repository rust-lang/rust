use ffi::{OsStr, CString};
use vec::Vec;
use libc;
use libc::consts::os::extra::ERROR_CALL_NOT_IMPLEMENTED;
use os::windows::prelude::*;
use ptr;
use sys::windows::c::{cvt, SetThreadErrorMode};
use sys::error::{self, Result};

pub type Error = error::Error;

pub const ENVVAR: &'static str = "PATH";
pub const SEPARATOR: &'static str = ";";

pub struct DynamicLibrary(*mut u8);

pub fn open(filename: Option<&OsStr>) -> Result<DynamicLibrary> {
    // disable "dll load failed" error dialog.
    let mut use_thread_mode = true;
    let prev_error_mode = unsafe {
        // SEM_FAILCRITICALERRORS 0x01
        let new_error_mode = 1;
        let mut prev_error_mode = 0;
        // Windows >= 7 supports thread error mode.
        let result = SetThreadErrorMode(new_error_mode, &mut prev_error_mode);
        if result == 0 {
            let err = error::expect_last_error();
            if err.code() as libc::c_int == ERROR_CALL_NOT_IMPLEMENTED {
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
                error::expect_last_result()
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
                error::expect_last_result()
            } else {
                Ok(handle as *mut u8)
            }
        }
    };

    unsafe {
        if use_thread_mode {
            SetThreadErrorMode(prev_error_mode, ptr::null_mut());
        } else {
            SetErrorMode(prev_error_mode);
        }
    }

    result.map(DynamicLibrary)
}

impl DynamicLibrary {
    pub fn symbol(&self, symbol: &str) -> Result<*mut u8> {
        let raw_string = CString::new(symbol).unwrap();
        unsafe {
            cvt(GetProcAddress(self.0 as *mut libc::c_void, raw_string.as_ptr()) as usize).map(|s| s as *mut u8)
        }
    }

    pub fn close(&self) -> Result<()> {
        unsafe {
            FreeLibrary(self.0 as *mut libc::c_void);
        }
        Ok(())
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
