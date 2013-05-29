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

Dynamic libraries.

Highly experimental, no guarantees if it works with non-Rust
libraries, or anything other than libraries compiled with the exact
Rust compiler that the program using this was compiled with.

*/

use str;
use cast;
use option::{None, Option};
use result::{Ok, Err, Result};
use libc::{c_int, c_void};
use ops::Drop;
use ptr::Ptr;

mod raw {
    use libc::{c_char, c_int, c_void};

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "android")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    pub enum RTLD {
        LAZY = 1,
        NOW = 2,
        GLOBAL = 256,
        LOCAL = 0
    }

    #[cfg(target_os = "linux")]
    #[cfg(target_os = "android")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    pub extern {
        fn dlopen(name: *c_char, mode: c_int) -> *c_void;
        fn dlsym(handle: *c_void, name: *c_char) -> *c_void;
        fn dlerror() -> *c_char;
        fn dlclose(handle: *c_void);
    }

    #[cfg(target_os = "win32")]
    pub extern {
        fn LoadLibrary(name: *c_char);
        fn GetProcAddress(handle: *c_void, name: *c_char) -> *c_void;
        fn FreeLibrary(handle: *c_void);
    }
}

/// An object representing a dynamic library
pub struct DynamicLib {
    priv handle: *c_void
}

/// An object representing a symbol from a dynamic library. A
/// work-around for either extern fns becoming &extern fn (so that
/// they can have a lifetime), or #5922.
pub struct TaggedSymbol<'self, T> {
    priv sym: T,
    priv lifetime: Option<&'self ()>
}

impl DynamicLib {
    /// Open a dynamic library.
    #[cfg(target_os = "linux")]
    #[cfg(target_os = "android")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    pub fn open(name: &str) -> Result<DynamicLib, ~str> {
        let handle = do str::as_c_str(name) |n| {
            unsafe { raw::dlopen(n, raw::NOW as c_int | raw::GLOBAL as c_int) }
        };
        if handle.is_not_null() {
            Ok(DynamicLib { handle: handle })
        } else {
            Err(
                unsafe {
                    let error = raw::dlerror();
                    if error.is_not_null() {
                        str::raw::from_c_str(error)
                    } else {
                        ~"unknown error"
                    }
                })
        }
    }
    /// Open a dynamic library.
    #[cfg(target_os = "win32")]
    pub fn open(name: &str) -> Result<DynamicLib, ~str> {
        let handle = do str::as_c_str(name) |n| {
            unsafe { raw::LoadLibrary(n) }
        };
        if handle.is_not_null() {
            Ok(DynamicLib { handle: handle })
        } else {
            let err = unsafe { intrinsics::GetLastError() } as uint;
            // XXX: make this message nice
            Err(fmt!("`LoadLibrary` failed with code %u", err))
        }
    }

    /// Retrieve a pointer to a symbol from the library. Note: this
    /// operation has no way of knowing if the symbol actually has
    /// type `T`, and so using the returned value can crash the
    /// program, not just cause the task to fail.
    #[cfg(target_os = "linux")]
    #[cfg(target_os = "android")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    pub fn get_symbol<'r, T>(&'r self, name: &str) -> Result<TaggedSymbol<'r, T>, ~str> {
        let sym = do str::as_c_str(name) |n| {
            unsafe { raw::dlsym(self.handle, n) }
        };
        if sym.is_not_null() {
            Ok(TaggedSymbol {
                sym: unsafe { cast::transmute(sym) },
                lifetime: None::<&'r ()>
            })
        } else {
            let error = unsafe { raw::dlerror() };
            if error.is_not_null() {
                Err(unsafe { str::raw::from_c_str(error) })
            } else {
                Err(~"unknown error")
            }
        }
    }

    /// Retrieve a pointer to a symbol from the library. Note: this
    /// operation has no way of knowing if the symbol actually has
    /// type `T`, and so using the returned value can crash the
    /// program, not just cause the task to fail.
    #[cfg(target_os = "win32")]
    pub fn get_symbol<'r, T>(&'r self, name: &str) -> Result<TaggedSymbol<'r, T>, ~str> {
        let sym = do str::as_c_str(name) |n| {
            unsafe { raw::GetProcAddress(self.handle, n) }
        };
        if sym.is_not_null() {
            Ok(TaggedSymbol { sym: unsafe { cast::transmute(sym) },
                             lifetime: None::<&'r ()> })
        } else {
            let err = unsafe { intrinsics::GetLastError() } as uint;
            Err(fmt!("`GetProcAddress` failed with code %u", err))
        }
    }
}

#[unsafe_destructor]
impl Drop for DynamicLib {
    #[cfg(target_os = "linux")]
    #[cfg(target_os = "android")]
    #[cfg(target_os = "macos")]
    #[cfg(target_os = "freebsd")]
    fn finalize(&self) {
        if self.handle.is_not_null() {
            unsafe { raw::dlclose(self.handle) }
        }
    }

    #[cfg(target_os = "win32")]
    fn finalize(&self) {
        if self.handle.is_not_null() {
            unsafe { raw::FreeLibrary(self.handle) }
        }
    }
}

impl<'self, T> TaggedSymbol<'self, T> {
    /// Get a reference to the (reference to the) symbol.
    ///
    /// WARNING: Using the returned value can lead to crashes, since
    /// the symbol may not really have type `T`.
    ///
    /// WARNING: Storing the derefence of this pointer can lead to
    /// crashes, if the library the symbol comes from has been dropped
    /// before using the stored symbol (the compiler doesn't give any
    /// indication that this is a problem).
    pub unsafe fn get<'r>(&'r self) -> &'r T {
        &'r self.sym
    }
}
