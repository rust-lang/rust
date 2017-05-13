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

use std::env;
use std::ffi::{CString, OsString};
use std::path::{Path, PathBuf};

pub struct DynamicLibrary {
    handle: *mut u8
}

impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        unsafe {
            dl::close(self.handle)
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
        let maybe_symbol_value = dl::symbol(self.handle, raw_string.as_ptr());

        // The value must not be constructed if there is an error so
        // the destructor does not run.
        match maybe_symbol_value {
            Err(err) => Err(err),
            Ok(symbol_value) => Ok(symbol_value as *mut T)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libc;
    use std::mem;

    #[test]
    fn test_loading_cosine() {
        if cfg!(windows) {
            return
        }

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
    fn test_errors_do_not_crash() {
        use std::path::Path;

        if !cfg!(unix) {
            return
        }

        // Open /dev/null as a library to get an error, and make sure
        // that only causes an error, and not a crash.
        let path = Path::new("/dev/null");
        match DynamicLibrary::open(Some(&path)) {
            Err(_) => {}
            Ok(_) => panic!("Successfully opened the empty library.")
        }
    }
}

#[cfg(unix)]
mod dl {
    use libc;
    use std::ffi::{CStr, OsStr, CString};
    use std::os::unix::prelude::*;
    use std::ptr;
    use std::str;

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
        let s = CString::new(filename.as_bytes()).unwrap();
        libc::dlopen(s.as_ptr(), LAZY) as *mut u8
    }

    unsafe fn open_internal() -> *mut u8 {
        libc::dlopen(ptr::null(), LAZY) as *mut u8
    }

    pub fn check_for_errors_in<T, F>(f: F) -> Result<T, String> where
        F: FnOnce() -> T,
    {
        use std::sync::{Mutex, Once, ONCE_INIT};
        static INIT: Once = ONCE_INIT;
        static mut LOCK: *mut Mutex<()> = 0 as *mut _;
        unsafe {
            INIT.call_once(|| {
                LOCK = Box::into_raw(Box::new(Mutex::new(())));
            });
            // dlerror isn't thread safe, so we need to lock around this entire
            // sequence
            let _guard = (*LOCK).lock();
            let _old_error = libc::dlerror();

            let result = f();

            let last_error = libc::dlerror() as *const _;
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
                         symbol: *const libc::c_char)
                         -> Result<*mut u8, String> {
        check_for_errors_in(|| {
            libc::dlsym(handle as *mut libc::c_void, symbol) as *mut u8
        })
    }
    pub unsafe fn close(handle: *mut u8) {
        libc::dlclose(handle as *mut libc::c_void); ()
    }
}

#[cfg(windows)]
mod dl {
    use std::ffi::OsStr;
    use std::io;
    use std::os::windows::prelude::*;
    use std::ptr;

    use libc::{c_uint, c_void, c_char};

    type DWORD = u32;
    type HMODULE = *mut u8;
    type BOOL = i32;
    type LPCWSTR = *const u16;
    type LPCSTR = *const i8;

    extern "system" {
        fn SetThreadErrorMode(dwNewMode: DWORD,
                              lpOldMode: *mut DWORD) -> c_uint;
        fn LoadLibraryW(name: LPCWSTR) -> HMODULE;
        fn GetModuleHandleExW(dwFlags: DWORD,
                              name: LPCWSTR,
                              handle: *mut HMODULE) -> BOOL;
        fn GetProcAddress(handle: HMODULE,
                          name: LPCSTR) -> *mut c_void;
        fn FreeLibrary(handle: HMODULE) -> BOOL;
    }

    pub fn open(filename: Option<&OsStr>) -> Result<*mut u8, String> {
        // disable "dll load failed" error dialog.
        let prev_error_mode = unsafe {
            // SEM_FAILCRITICALERRORS 0x01
            let new_error_mode = 1;
            let mut prev_error_mode = 0;
            let result = SetThreadErrorMode(new_error_mode,
                                            &mut prev_error_mode);
            if result == 0 {
                return Err(io::Error::last_os_error().to_string())
            }
            prev_error_mode
        };

        let result = match filename {
            Some(filename) => {
                let filename_str: Vec<_> =
                    filename.encode_wide().chain(Some(0)).collect();
                let result = unsafe {
                    LoadLibraryW(filename_str.as_ptr())
                };
                ptr_result(result)
            }
            None => {
                let mut handle = ptr::null_mut();
                let succeeded = unsafe {
                    GetModuleHandleExW(0 as DWORD, ptr::null(), &mut handle)
                };
                if succeeded == 0 {
                    Err(io::Error::last_os_error().to_string())
                } else {
                    Ok(handle as *mut u8)
                }
            }
        };

        unsafe {
            SetThreadErrorMode(prev_error_mode, ptr::null_mut());
        }

        result
    }

    pub unsafe fn symbol(handle: *mut u8,
                         symbol: *const c_char)
                         -> Result<*mut u8, String> {
        let ptr = GetProcAddress(handle as HMODULE, symbol) as *mut u8;
        ptr_result(ptr)
    }

    pub unsafe fn close(handle: *mut u8) {
        FreeLibrary(handle as HMODULE);
    }

    fn ptr_result<T>(ptr: *mut T) -> Result<*mut T, String> {
        if ptr.is_null() {
            Err(io::Error::last_os_error().to_string())
        } else {
            Ok(ptr)
        }
    }
}
