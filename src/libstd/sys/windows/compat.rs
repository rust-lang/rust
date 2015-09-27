// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A "compatibility layer" for spanning XP and Windows 7
//!
//! The standard library currently binds many functions that are not available
//! on Windows XP, but we would also like to support building executables that
//! run on XP. To do this we specify all non-XP APIs as having a fallback
//! implementation to do something reasonable.
//!
//! This dynamic runtime detection of whether a function is available is
//! implemented with `GetModuleHandle` and `GetProcAddress` paired with a
//! static-per-function which caches the result of the first check. In this
//! manner we pay a semi-large one-time cost up front for detecting whether a
//! function is available but afterwards it's just a load and a jump.

use prelude::v1::*;

use ffi::CString;
use libc::{LPVOID, LPCWSTR, HMODULE, LPCSTR};
use sync::atomic::{AtomicPtr, Ordering};

extern "system" {
    fn GetModuleHandleW(lpModuleName: LPCWSTR) -> HMODULE;
    fn GetProcAddress(hModule: HMODULE, lpProcName: LPCSTR) -> LPVOID;
}

pub fn lookup(module: &str, symbol: &str) -> Option<*mut ()> {
    let mut module: Vec<u16> = module.utf16_units().collect();
    module.push(0);
    let symbol = CString::new(symbol).unwrap();
    unsafe {
        let handle = GetModuleHandleW(module.as_ptr());
        match GetProcAddress(handle, symbol.as_ptr()) as usize {
            0 => None,
            n => Some(n as *mut ()),
        }
    }
}

pub fn store_func(ptr: &AtomicPtr<()>, module: &str, symbol: &str,
                  fallback: *mut ()) -> *mut () {
    let value = lookup(module, symbol).unwrap_or(fallback);
    ptr.store(value, Ordering::Relaxed);
    value
}

macro_rules! compat_fn {
    ($module:ident: $(
        pub fn $symbol:ident($($argname:ident: $argtype:ty),*)
                                  -> $rettype:ty {
            $($body:expr);*
        }
    )*) => ($(
        #[inline]
        pub unsafe fn $symbol($($argname: $argtype),*) -> $rettype {
            use sync::atomic::{AtomicPtr, Ordering};
            use mem;
            type F = unsafe extern "system" fn($($argtype),*) -> $rettype;

            static PTR: AtomicPtr<()> = AtomicPtr::new(load as *mut ());

            unsafe extern "system" fn load($($argname: $argtype),*)
                                           -> $rettype {
                let ptr = ::sys::compat::store_func(&PTR,
                                                    stringify!($module),
                                                    stringify!($symbol),
                                                    fallback as *mut ());
                mem::transmute::<*mut (), F>(ptr)($($argname),*)
            }

            #[allow(unused_variables)]
            unsafe extern "system" fn fallback($($argname: $argtype),*)
                                               -> $rettype {
                $($body);*
            }

            let ptr = PTR.load(Ordering::Relaxed);
            mem::transmute::<*mut (), F>(ptr)($($argname),*)
        }
    )*)
}

macro_rules! compat_group {
    ($gtype:ident, $gstatic:ident, $gload:ident, $module:ident: $(
        pub fn $symbol:ident($($argname:ident: $argtype:ty),*)
                                  -> $rettype:ty {
            $($body:expr);*
        }
    )*) => (
        struct $gtype {
            $($symbol: ::sync::atomic::AtomicPtr<()>),*
        }
        static $gstatic: $gtype = $gtype {$(
            $symbol: ::sync::atomic::AtomicPtr::new({
                type F = unsafe extern "system" fn($($argtype),*) -> $rettype;
                unsafe extern "system" fn $symbol($($argname: $argtype),*)
                                                  -> $rettype {
                    use self::$symbol;
                    $gload();
                    $symbol($($argname),*)
                }
                $symbol as *mut ()
            })
        ),*};

        fn $gload() {
            use option::Option::{None, Some};
            use sync::atomic::Ordering;
            use ptr;
            $(
                #[allow(unused_variables)]
                unsafe extern "system" fn $symbol($($argname: $argtype),*)
                                                  -> $rettype {
                    $($body);*
                }
            )*

            struct FuncPtrs {
                $($symbol: *mut ()),*
            }

            const FALLBACKS: FuncPtrs = FuncPtrs {
                $($symbol: $symbol as *mut ()),*
            };

            fn store_funcs(funcs: &FuncPtrs) {
                $($gstatic.$symbol.store(funcs.$symbol, Ordering::Relaxed);)*
            }

            let mut funcs: FuncPtrs = FuncPtrs {
                $($symbol: ptr::null_mut()),*
            };

            $(
                let ptr = ::sys::compat::lookup(stringify!($module), stringify!($symbol));
                match ptr {
                    Some(ptr) => { funcs.$symbol = ptr; },
                    None => {
                        store_funcs(&FALLBACKS);
                        return;
                    }
                }
            )*

            store_funcs(&funcs);
        }

        $(
            #[inline]
            pub unsafe fn $symbol($($argname: $argtype),*) -> $rettype {
                use sync::atomic::Ordering;
                use mem;
                type F = unsafe extern "system" fn($($argtype),*) -> $rettype;
                let ptr = $gstatic.$symbol.load(Ordering::Relaxed);
                mem::transmute::<*mut (), F>(ptr)($($argname),*)
            }
        )*
    )
}
