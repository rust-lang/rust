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

use crate::ffi::CString;
use crate::sys::c;

pub fn lookup(module: &str, symbol: &str) -> Option<usize> {
    let mut module: Vec<u16> = module.encode_utf16().collect();
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

macro_rules! compat_fn {
    ($module:literal: $(
        $(#[$meta:meta])*
        pub fn $symbol:ident($($argname:ident: $argtype:ty),*) -> $rettype:ty $body:block
    )*) => ($(
        $(#[$meta])*
        pub mod $symbol {
            #[allow(unused_imports)]
            use super::*;
            use crate::sync::atomic::{AtomicUsize, Ordering};
            use crate::mem;

            type F = unsafe extern "system" fn($($argtype),*) -> $rettype;

            static PTR: AtomicUsize = AtomicUsize::new(0);

            #[allow(unused_variables)]
            unsafe extern "system" fn fallback($($argname: $argtype),*) -> $rettype $body

            /// This address is stored in `PTR` to incidate an unavailable API.
            ///
            /// This way, call() will end up calling fallback() if it is unavailable.
            ///
            /// This is a `static` to avoid rustc duplicating `fn fallback()`
            /// into both load() and is_available(), which would break
            /// is_available()'s comparison. By using the same static variable
            /// in both places, they'll refer to the same (copy of the)
            /// function.
            ///
            /// LLVM merging the address of fallback with other functions
            /// (because of unnamed_addr) is fine, since it's only compared to
            /// an address from GetProcAddress from an external dll.
            static FALLBACK: F = fallback;

            #[cold]
            fn load() -> usize {
                // There is no locking here. It's okay if this is executed by multiple threads in
                // parallel. `lookup` will result in the same value, and it's okay if they overwrite
                // eachothers result as long as they do so atomically. We don't need any guarantees
                // about memory ordering, as this involves just a single atomic variable which is
                // not used to protect or order anything else.
                let addr = crate::sys::compat::lookup($module, stringify!($symbol))
                    .unwrap_or(FALLBACK as usize);
                PTR.store(addr, Ordering::Relaxed);
                addr
            }

            fn addr() -> usize {
                match PTR.load(Ordering::Relaxed) {
                    0 => load(),
                    addr => addr,
                }
            }

            #[allow(dead_code)]
            pub fn is_available() -> bool {
                addr() != FALLBACK as usize
            }

            pub unsafe fn call($($argname: $argtype),*) -> $rettype {
                mem::transmute::<usize, F>(addr())($($argname),*)
            }
        }

        $(#[$meta])*
        pub use $symbol::call as $symbol;
    )*)
}
