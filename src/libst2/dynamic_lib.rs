// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/*!

Dynamic library facilities.

A simple wrapper over the platform's dynamic library facilities

*/

#![experimental]
#![allow(missing_docs)]

use clone::Clone;
use c_str::ToCStr;
use iter::Iterator;
use mem;
use ops::*;
use option::*;
use os;
use path::{Path,GenericPath};
use result::*;
use slice::{AsSlice,SlicePrelude};
use str;
use string::String;
use vec::Vec;

pub struct DynamicLibrary { handle: *mut u8 }

impl Drop for DynamicLibrary {
    fn drop(&mut self) { unimplemented!() }
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
    pub fn open<T: ToCStr>(filename: Option<T>)
                        -> Result<DynamicLibrary, String> { unimplemented!() }

    /// Prepends a path to this process's search path for dynamic libraries
    pub fn prepend_search_path(path: &Path) { unimplemented!() }

    /// From a slice of paths, create a new vector which is suitable to be an
    /// environment variable for this platforms dylib search path.
    pub fn create_path(path: &[Path]) -> Vec<u8> { unimplemented!() }

    /// Returns the environment variable for this process's dynamic library
    /// search path
    pub fn envvar() -> &'static str { unimplemented!() }

    fn separator() -> u8 { unimplemented!() }

    /// Returns the current search path for dynamic libraries being used by this
    /// process
    pub fn search_path() -> Vec<Path> { unimplemented!() }

    /// Access the value at the symbol of the dynamic library
    pub unsafe fn symbol<T>(&self, symbol: &str) -> Result<*mut T, String> { unimplemented!() }
}

#[cfg(any(target_os = "linux",
          target_os = "android",
          target_os = "macos",
          target_os = "ios",
          target_os = "freebsd",
          target_os = "dragonfly"))]
pub mod dl {
    pub use self::Rtld::*;

    use c_str::{CString, ToCStr};
    use libc;
    use ptr;
    use result::*;
    use string::String;

    pub unsafe fn open_external<T: ToCStr>(filename: T) -> *mut u8 { unimplemented!() }

    pub unsafe fn open_internal() -> *mut u8 { unimplemented!() }

    pub fn check_for_errors_in<T>(f: || -> T) -> Result<T, String> { unimplemented!() }

    pub unsafe fn symbol(handle: *mut u8,
                         symbol: *const libc::c_char) -> *mut u8 { unimplemented!() }
    pub unsafe fn close(handle: *mut u8) { unimplemented!() }

    pub enum Rtld {
        Lazy = 1,
        Now = 2,
        Global = 256,
        Local = 0,
    }

    #[link_name = "dl"]
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
pub mod dl {
    use c_str::ToCStr;
    use iter::Iterator;
    use libc;
    use os;
    use ptr;
    use result::{Ok, Err, Result};
    use slice::SlicePrelude;
    use str::StrPrelude;
    use str;
    use string::String;
    use vec::Vec;

    pub unsafe fn open_external<T: ToCStr>(filename: T) -> *mut u8 { unimplemented!() }

    pub unsafe fn open_internal() -> *mut u8 { unimplemented!() }

    pub fn check_for_errors_in<T>(f: || -> T) -> Result<T, String> { unimplemented!() }

    pub unsafe fn symbol(handle: *mut u8, symbol: *const libc::c_char) -> *mut u8 { unimplemented!() }
    pub unsafe fn close(handle: *mut u8) { unimplemented!() }

    #[allow(non_snake_case)]
    extern "system" {
        fn SetLastError(error: libc::size_t);
        fn LoadLibraryW(name: *const libc::c_void) -> *mut libc::c_void;
        fn GetModuleHandleExW(dwFlags: libc::DWORD, name: *const u16,
                              handle: *mut *mut libc::c_void)
                              -> *mut libc::c_void;
        fn GetProcAddress(handle: *mut libc::c_void,
                          name: *const libc::c_char) -> *mut libc::c_void;
        fn FreeLibrary(handle: *mut libc::c_void);
    }
}
