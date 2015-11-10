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
use sync::atomic::{AtomicUsize, Ordering};
use sys::c;

pub fn lookup(module: &str, symbol: &str) -> Option<usize> {
    let mut module: Vec<u16> = module.utf16_units().collect();
    module.push(0);
    let symbol = CString::new(symbol).unwrap();
    unsafe {
        let handle = c::GetModuleHandleW(module.as_ptr());
        match c::GetProcAddress(handle, symbol.as_ptr()) as usize {
            0 => None,
            n => Some(n),
        }
    }
}

pub fn store_func(ptr: &AtomicUsize, module: &str, symbol: &str,
                  fallback: usize) -> usize {
    let value = lookup(module, symbol).unwrap_or(fallback);
    ptr.store(value, Ordering::SeqCst);
    value
}

macro_rules! compat_fn {
    ($module:ident: $(
        pub fn $symbol:ident($($argname:ident: $argtype:ty),*)
                                  -> $rettype:ty {
            $($body:expr);*
        }
    )*) => ($(
        #[allow(unused_variables)]
        pub unsafe fn $symbol($($argname: $argtype),*) -> $rettype {
            use sync::atomic::{AtomicUsize, Ordering};
            use mem;
            type F = unsafe extern "system" fn($($argtype),*) -> $rettype;

            static PTR: AtomicUsize = AtomicUsize::new(0);

            fn load() -> usize {
                ::sys::compat::store_func(&PTR,
                                          stringify!($module),
                                          stringify!($symbol),
                                          fallback as usize)
            }
            unsafe extern "system" fn fallback($($argname: $argtype),*)
                                               -> $rettype {
                $($body);*
            }

            let addr = match PTR.load(Ordering::SeqCst) {
                0 => load(),
                n => n,
            };
            mem::transmute::<usize, F>(addr)($($argname),*)
        }
    )*)
}
