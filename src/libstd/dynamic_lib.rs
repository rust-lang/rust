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
#![deprecated(since = "1.5.0", reason = "replaced with crates.io crates")]
#![allow(missing_docs)]
#![allow(deprecated)]

use prelude::v1::*;
use sys::dynamic_lib as sys;

use env;
use ffi::OsString;
use path::{Path, PathBuf};

pub struct DynamicLibrary {
    handle: sys::DynamicLibrary
}

impl Drop for DynamicLibrary {
    fn drop(&mut self) {
        match sys::DynamicLibrary::close(&self.handle) {
            Ok(()) => {},
            Err(str) => panic!("{}", str)
        }
    }
}

impl DynamicLibrary {
    /// Lazily open a dynamic library. When passed None it gives a
    /// handle to the calling process
    pub fn open(filename: Option<&Path>) -> Result<DynamicLibrary, sys::Error> {
        let maybe_library = sys::open(filename.map(Path::as_os_str));

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
        sys::ENVVAR
    }

    fn separator() -> &'static str {
        sys::SEPARATOR
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
    pub unsafe fn symbol<T>(&self, symbol: &str) -> Result<*mut T, sys::Error> {
        // This function should have a lifetime constraint of 'a on
        // T but that feature is still unimplemented

        let maybe_symbol_value = self.handle.symbol(symbol);

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
    use sys::c::prelude as c;
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

        let cosine: extern fn(c::c_double) -> c::c_double = unsafe {
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
