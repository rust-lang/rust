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

#![unstable(feature = "std_misc")]
#![allow(missing_docs)]

use prelude::v1::*;

use ffi::CString;
use mem;
use env;
use str;

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
    // FIXME (#12938): Until DST lands, we cannot decompose &str into
    // & and str, so we cannot usefully take ToCStr arguments by
    // reference (without forcing an additional & around &str). So we
    // are instead temporarily adding an instance for &Path, so that
    // we can take ToCStr as owned. When DST lands, the &Path instance
    // should be removed, and arguments bound by ToCStr should be
    // passed by reference. (Here: in the `open` method.)

    /// Lazily open a dynamic library. When passed None it gives a
    /// handle to the calling process
    pub fn open(filename: Option<&Path>) -> Result<DynamicLibrary, String> {
        let maybe_library = dl::open(filename.map(|path| path.as_vec()));

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
        search_path.insert(0, path.clone());
        let newval = DynamicLibrary::create_path(&search_path);
        env::set_var(DynamicLibrary::envvar(),
                     str::from_utf8(&newval).unwrap());
    }

    /// From a slice of paths, create a new vector which is suitable to be an
    /// environment variable for this platforms dylib search path.
    pub fn create_path(path: &[Path]) -> Vec<u8> {
        let mut newvar = Vec::new();
        for (i, path) in path.iter().enumerate() {
            if i > 0 { newvar.push(DynamicLibrary::separator()); }
            newvar.push_all(path.as_vec());
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

    fn separator() -> u8 {
        if cfg!(windows) {b';'} else {b':'}
    }

    /// Returns the current search path for dynamic libraries being used by this
    /// process
    pub fn search_path() -> Vec<Path> {
        match env::var_os(DynamicLibrary::envvar()) {
            Some(var) => env::split_paths(&var).collect(),
            None => Vec::new(),
        }
    }

    /// Access the value at the symbol of the dynamic library
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
            Ok(symbol_value) => Ok(mem::transmute(symbol_value))
        }
    }
}

#[cfg(all(test, not(target_os = "ios")))]
mod test {
    use super::*;
    use prelude::v1::*;
    use libc;
    use mem;

    #[test]
    #[cfg_attr(any(windows, target_os = "android"), ignore)] // FIXME #8818, #10379
    fn test_loading_cosine() {
        // The math library does not need to be loaded since it is already
        // statically linked in
        let none: Option<&Path> = None; // appease the typechecker
        let libm = match DynamicLibrary::open(none) {
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
          target_os = "openbsd"))]
mod dl {
    use prelude::v1::*;

    use ffi::{CString, CStr};
    use str;
    use libc;
    use ptr;

    pub fn open(filename: Option<&[u8]>) -> Result<*mut u8, String> {
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

    unsafe fn open_external(filename: &[u8]) -> *mut u8 {
        let s = CString::new(filename).unwrap();
        dlopen(s.as_ptr(), LAZY) as *mut u8
    }

    unsafe fn open_internal() -> *mut u8 {
        dlopen(ptr::null(), LAZY) as *mut u8
    }

    pub fn check_for_errors_in<T, F>(f: F) -> Result<T, String> where
        F: FnOnce() -> T,
    {
        use sync::{StaticMutex, MUTEX_INIT};
        static LOCK: StaticMutex = MUTEX_INIT;
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
                Err(str::from_utf8(s).unwrap().to_string())
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
    use iter::IteratorExt;
    use libc;
    use libc::consts::os::extra::ERROR_CALL_NOT_IMPLEMENTED;
    use ops::FnOnce;
    use os;
    use option::Option::{self, Some, None};
    use ptr;
    use result::Result;
    use result::Result::{Ok, Err};
    use slice::SliceExt;
    use str::StrExt;
    use str;
    use string::String;
    use vec::Vec;
    use sys::c::compat::kernel32::SetThreadErrorMode;

    pub fn open(filename: Option<&[u8]>) -> Result<*mut u8, String> {
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
                    // SetThreadErrorMode not found. use fallback solution: SetErrorMode()
                    // Note that SetErrorMode is process-wide so this can cause race condition!
                    // However, since even Windows APIs do not care of such problem (#20650),
                    // we just assume SetErrorMode race is not a great deal.
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
                let filename_str = str::from_utf8(filename).unwrap();
                let mut filename_str: Vec<u16> = filename_str.utf16_units().collect();
                filename_str.push(0);
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
