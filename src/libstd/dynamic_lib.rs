// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Dynamic library facilities.
//!
//! A simple wrapper over the platform's dynamic library facilities

#![unstable(feature = "dynamic_lib",
            reason = "API has not been scrutinized and is highly likely to \
                      either disappear or change",
            issue = "27810")]
#![allow(missing_docs)]

use prelude::v1::*;

use env;
use ffi::{CString, OsString};
use path::{Path, PathBuf};

pub struct DynamicLibrary {
    handle: *mut u8
}

impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        match dl::check_for_errors_in(|| {
            unsafe {
                dl::close(self.handle)
            }
        }) {
            Ok(()) => {},
            Err(str) => panic!("{}", str)
        }
    }
}

impl DynamicLibrary {
    /// Lazily open a dynamic library. When passed None it gives a
    /// handle to the calling process
    pub fn open(filename: Option<&Path>) -> Result<DynamicLibrary, String> {
        let maybe_library = dl::open(filename.map(|path| path.as_os_str()));

        // The dynamic library must not be constructed if there is
        // an error opening the library so the destructor does not
        // run.
        match maybe_library {
            Err(err) => Err(err),
            Ok(handle) => Ok(DynamicLibrary { handle: handle })
        }
    }

    /// Prepends a path to this process's search path for dynamic libraries
    pub fn prepend_search_path(path: &Path) {
        let mut search_path = DynamicLibrary::search_path();
        search_path.insert(0, path.to_path_buf());
        env::set_var(DynamicLibrary::envvar(), &DynamicLibrary::create_path(&search_path));
    }

    /// From a slice of paths, create a new vector which is suitable to be an
    /// environment variable for this platforms dylib search path.
    pub fn create_path(path: &[PathBuf]) -> OsString {
        let mut newvar = OsString::new();
        for (i, path) in path.iter().enumerate() {
            if i > 0 { newvar.push(DynamicLibrary::separator()); }
            newvar.push(path);
        }
        return newvar;
    }

    /// Returns the environment variable for this process's dynamic library
    /// search path
    pub fn envvar() -> &'static str {
        if cfg!(windows) {
            "PATH"
        } else if cfg!(target_os = "macos") {
            "DYLD_LIBRARY_PATH"
        } else {
            "LD_LIBRARY_PATH"
        }
    }

    fn separator() -> &'static str {
        if cfg!(windows) { ";" } else { ":" }
    }

    /// Returns the current search path for dynamic libraries being used by this
    /// process
    pub fn search_path() -> Vec<PathBuf> {
        match env::var_os(DynamicLibrary::envvar()) {
            Some(var) => env::split_paths(&var).collect(),
            None => Vec::new(),
        }
    }

    /// Accesses the value at the symbol of the dynamic library.
    pub unsafe fn symbol<T>(&self, symbol: &str) -> Result<*mut T, String> {
        // This function should have a lifetime constraint of 'a on
        // T but that feature is still unimplemented

        let raw_string = CString::new(symbol).unwrap();
        let maybe_symbol_value = dl::check_for_errors_in(|| {
            dl::symbol(self.handle, raw_string.as_ptr())
        });

        // The value must not be constructed if there is an error so
        // the destructor does not run.
        match maybe_symbol_value {
            Err(err) => Err(err),
            Ok(symbol_value) => Ok(symbol_value as *mut T)
        }
    }
}

#[cfg(all(test, not(target_os = "ios"), not(target_os = "nacl")))]
mod tests {
    use super::*;
    use prelude::v1::*;
    use libc;
    use mem;
    use path::Path;

    #[test]
    #[cfg_attr(any(windows,
                   target_os = "android",  // FIXME #10379
                   target_env = "musl"), ignore)]
    fn test_loading_cosine() {
        // The math library does not need to be loaded since it is already
        // statically linked in
        let libm = match DynamicLibrary::open(None) {
            Err(error) => panic!("Could not load self as module: {}", error),
            Ok(libm) => libm
        };

        let cosine: extern fn(libc::c_double) -> libc::c_double = unsafe {
            match libm.symbol("cos") {
                Err(error) => panic!("Could not load function cos: {}", error),
                Ok(cosine) => mem::transmute::<*mut u8, _>(cosine)
            }
        };

        let argument = 0.0;
        let expected_result = 1.0;
        let result = cosine(argument);
        if result != expected_result {
            panic!("cos({}) != {} but equaled {} instead", argument,
                   expected_result, result)
        }
    }

    #[test]
    #[cfg(any(target_os = "linux",
              target_os = "macos",
              target_os = "freebsd",
              target_os = "dragonfly",
              target_os = "bitrig",
              target_os = "netbsd",
              target_os = "openbsd"))]
    fn test_errors_do_not_crash() {
        // Open /dev/null as a library to get an error, and make sure
        // that only causes an error, and not a crash.
        let path = Path::new("/dev/null");
        match DynamicLibrary::open(Some(&path)) {
            Err(_) => {}
            Ok(_) => panic!("Successfully opened the empty library.")
        }
    }
}

#[cfg(any(target_os = "linux",
          target_os = "android",
          target_os = "macos",
          target_os = "ios",
          target_os = "freebsd",
          target_os = "dragonfly",
          target_os = "bitrig",
          target_os = "netbsd",
          target_os = "openbsd"))]
mod dl {
    use prelude::v1::*;

    use ffi::{CStr, OsStr};
    use str;
    use libc;
    use ptr;

    pub fn open(filename: Option<&OsStr>) -> Result<*mut u8, String> {
        check_for_errors_in(|| {
            unsafe {
                match filename {
                    Some(filename) => open_external(filename),
                    None => open_internal(),
                }
            }
        })
    }

    const LAZY: libc::c_int = 1;

    unsafe fn open_external(filename: &OsStr) -> *mut u8 {
        let s = filename.to_cstring().unwrap();
        dlopen(s.as_ptr(), LAZY) as *mut u8
    }

    unsafe fn open_internal() -> *mut u8 {
        dlopen(ptr::null(), LAZY) as *mut u8
    }

    pub fn check_for_errors_in<T, F>(f: F) -> Result<T, String> where
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

    pub unsafe fn symbol(handle: *mut u8,
                         symbol: *const libc::c_char) -> *mut u8 {
        dlsym(handle as *mut libc::c_void, symbol) as *mut u8
    }
    pub unsafe fn close(handle: *mut u8) {
        dlclose(handle as *mut libc::c_void); ()
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

#[cfg(target_os = "windows")]
mod dl {
    use prelude::v1::*;

    use ffi::OsStr;
    use libc;
    use libc::consts::os::extra::ERROR_CALL_NOT_IMPLEMENTED;
    use sys::os;
    use os::windows::prelude::*;
    use ptr;
    use sys::c::SetThreadErrorMode;

    pub fn open(filename: Option<&OsStr>) -> Result<*mut u8, String> {
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
        };

        unsafe {
            if use_thread_mode {
                SetThreadErrorMode(prev_error_mode, ptr::null_mut());
            } else {
                SetErrorMode(prev_error_mode);
            }
        }

        result
    }

    pub fn check_for_errors_in<T, F>(f: F) -> Result<T, String> where
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

    pub unsafe fn symbol(handle: *mut u8, symbol: *const libc::c_char) -> *mut u8 {
        GetProcAddress(handle as *mut libc::c_void, symbol) as *mut u8
    }
    pub unsafe fn close(handle: *mut u8) {
        FreeLibrary(handle as *mut libc::c_void); ()
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

#[cfg(target_os = "nacl")]
pub mod dl {
    use ffi::OsStr;
    use ptr;
    use result::Result;
    use result::Result::Err;
    use libc;
    use string::String;
    use ops::FnOnce;
    use option::Option;

    pub fn open(_filename: Option<&OsStr>) -> Result<*mut u8, String> {
        Err(format!("NaCl + Newlib doesn't impl loading shared objects"))
    }

    pub fn check_for_errors_in<T, F>(_f: F) -> Result<T, String>
        where F: FnOnce() -> T,
    {
        Err(format!("NaCl doesn't support shared objects"))
    }

    pub unsafe fn symbol(_handle: *mut u8, _symbol: *const libc::c_char) -> *mut u8 {
        ptr::null_mut()
    }
    pub unsafe fn close(_handle: *mut u8) { }
}
