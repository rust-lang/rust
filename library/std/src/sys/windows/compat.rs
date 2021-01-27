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

macro_rules! compat_fn {
    ($module:literal: $(
        $(#[$meta:meta])*
        pub fn $symbol:ident($($argname:ident: $argtype:ty),*) -> $rettype:ty $body:block
    )*) => ($(
        $(#[$meta])*
        pub mod $symbol {
            #[allow(unused_imports)]
            use super::*;
            use crate::mem;

            type F = unsafe extern "system" fn($($argtype),*) -> $rettype;

            static mut PTR: F = fallback;

            #[allow(unused_variables)]
            unsafe extern "system" fn fallback($($argname: $argtype),*) -> $rettype $body

            /// This address is stored in `PTR` to indicate an unavailable API.
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

            #[used]
            #[link_section = ".CRT$XCU"]
            static XCU_ENTRY: fn() = init_me;

            #[cold]
            fn init_me() {
                // There is no locking here. This code is executed before main() is entered, and
                // is guaranteed to be single-threaded.
                unsafe {
                    let module_name: *const u8 = concat!($module, "\0").as_ptr();
                    let symbol_name: *const u8 = concat!(stringify!($symbol), "\0").as_ptr();
                    let module_handle = $crate::sys::c::GetModuleHandleA(module_name as *const i8);
                    if !module_handle.is_null() {
                        match $crate::sys::c::GetProcAddress(module_handle, symbol_name as *const i8) as usize {
                            0 => {}
                            n => {
                                PTR = mem::transmute::<usize, F>(n);
                            }
                        }
                    }
                }
            }

            #[allow(dead_code)]
            pub fn is_available() -> bool {
                unsafe { PTR as usize != FALLBACK as usize }
            }

            pub unsafe fn call($($argname: $argtype),*) -> $rettype {
                (PTR)($($argname),*)
            }
        }

        $(#[$meta])*
        pub use $symbol::call as $symbol;
    )*)
}
