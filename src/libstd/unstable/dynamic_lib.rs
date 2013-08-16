// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
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

A simple wrapper over the platforms dynamic library facilities

*/
use c_str::ToCStr;
use cast;
use path;
use libc;
use ops::*;
use option::*;
use result::*;

pub struct DynamicLibrary { priv handle: *libc::c_void }

impl Drop for DynamicLibrary {
    fn drop(&self) {
        match do dl::check_for_errors_in {
            unsafe {
                dl::close(self.handle)
            }
        } {
            Ok(()) => {},
            Err(str) => fail!(str)
        }
    }
}

impl DynamicLibrary {
    /// Lazily open a dynamic library. When passed None it gives a
    /// handle to the calling process
    pub fn open(filename: Option<&path::Path>) -> Result<DynamicLibrary, ~str> {
        unsafe {
            let maybe_library = do dl::check_for_errors_in {
                match filename {
                    Some(name) => dl::open_external(name),
                    None => dl::open_internal()
                }
            };

            // The dynamic library must not be constructed if there is
            // an error opening the library so the destructor does not
            // run.
            match maybe_library {
                Err(err) => Err(err),
                Ok(handle) => Ok(DynamicLibrary { handle: handle })
            }
        }
    }

    /// Access the value at the symbol of the dynamic library
    pub unsafe fn symbol<T>(&self, symbol: &str) -> Result<T, ~str> {
        // This function should have a lifetime constraint of 'self on
        // T but that feature is still unimplemented

        let maybe_symbol_value = do dl::check_for_errors_in {
            do symbol.with_c_str |raw_string| {
                dl::symbol(self.handle, raw_string)
            }
        };

        // The value must not be constructed if there is an error so
        // the destructor does not run.
        match maybe_symbol_value {
            Err(err) => Err(err),
            Ok(symbol_value) => Ok(cast::transmute(symbol_value))
        }
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use option::*;
    use result::*;
    use path::*;
    use libc;

    #[test]
    fn test_loading_cosine() {
        // The math library does not need to be loaded since it is already
        // statically linked in
        let libm = match DynamicLibrary::open(None) {
            Err(error) => fail!("Could not load self as module: %s", error),
            Ok(libm) => libm
        };

        // Unfortunately due to issue #6194 it is not possible to call
        // this as a C function
        let cosine: extern fn(libc::c_double) -> libc::c_double = unsafe {
            match libm.symbol("cos") {
                Err(error) => fail!("Could not load function cos: %s", error),
                Ok(cosine) => cosine
            }
        };

        let argument = 0.0;
        let expected_result = 1.0;
        let result = cosine(argument);
        if result != expected_result {
            fail!("cos(%?) != %? but equaled %? instead", argument,
                  expected_result, result)
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn test_errors_do_not_crash() {
        // Open /dev/null as a library to get an error, and make sure
        // that only causes an error, and not a crash.
        let path = GenericPath::from_str("/dev/null");
        match DynamicLibrary::open(Some(&path)) {
            Err(_) => {}
            Ok(_) => fail!("Successfully opened the empty library.")
        }
    }
}

#[cfg(target_os = "linux")]
#[cfg(target_os = "android")]
#[cfg(target_os = "macos")]
#[cfg(target_os = "freebsd")]
mod dl {
    use c_str::ToCStr;
    use libc;
    use path;
    use ptr;
    use str;
    use unstable::sync::atomically;
    use result::*;

    pub unsafe fn open_external(filename: &path::Path) -> *libc::c_void {
        do filename.with_c_str |raw_name| {
            dlopen(raw_name, Lazy as libc::c_int)
        }
    }

    pub unsafe fn open_internal() -> *libc::c_void {
        dlopen(ptr::null(), Lazy as libc::c_int)
    }

    pub fn check_for_errors_in<T>(f: &fn()->T) -> Result<T, ~str> {
        unsafe {
            do atomically {
                let _old_error = dlerror();

                let result = f();

                let last_error = dlerror();
                if ptr::null() == last_error {
                    Ok(result)
                } else {
                    Err(str::raw::from_c_str(last_error))
                }
            }
        }
    }

    pub unsafe fn symbol(handle: *libc::c_void, symbol: *libc::c_char) -> *libc::c_void {
        dlsym(handle, symbol)
    }
    pub unsafe fn close(handle: *libc::c_void) {
        dlclose(handle); ()
    }

    pub enum RTLD {
        Lazy = 1,
        Now = 2,
        Global = 256,
        Local = 0,
    }

    #[link_name = "dl"]
    extern {
        fn dlopen(filename: *libc::c_char, flag: libc::c_int) -> *libc::c_void;
        fn dlerror() -> *libc::c_char;
        fn dlsym(handle: *libc::c_void, symbol: *libc::c_char) -> *libc::c_void;
        fn dlclose(handle: *libc::c_void) -> libc::c_int;
    }
}

#[cfg(target_os = "win32")]
mod dl {
    use os;
    use libc;
    use path;
    use ptr;
    use unstable::sync::atomically;
    use result::*;

    pub unsafe fn open_external(filename: &path::Path) -> *libc::c_void {
        do os::win32::as_utf16_p(filename.to_str()) |raw_name| {
            LoadLibraryW(raw_name)
        }
    }

    pub unsafe fn open_internal() -> *libc::c_void {
        let handle = ptr::null();
        GetModuleHandleExW(0 as libc::DWORD, ptr::null(), &handle as **libc::c_void);
        handle
    }

    pub fn check_for_errors_in<T>(f: &fn()->T) -> Result<T, ~str> {
        unsafe {
            do atomically {
                SetLastError(0);

                let result = f();

                let error = os::errno();
                if 0 == error {
                    Ok(result)
                } else {
                    Err(fmt!("Error code %?", error))
                }
            }
        }
    }
    pub unsafe fn symbol(handle: *libc::c_void, symbol: *libc::c_char) -> *libc::c_void {
        GetProcAddress(handle, symbol)
    }
    pub unsafe fn close(handle: *libc::c_void) {
        FreeLibrary(handle); ()
    }

    #[link_name = "kernel32"]
    extern "stdcall" {
        fn SetLastError(error: u32);
        fn LoadLibraryW(name: *u16) -> *libc::c_void;
        fn GetModuleHandleExW(dwFlags: libc::DWORD, name: *u16,
                              handle: **libc::c_void) -> *libc::c_void;
        fn GetProcAddress(handle: *libc::c_void, name: *libc::c_char) -> *libc::c_void;
        fn FreeLibrary(handle: *libc::c_void);
    }
}
