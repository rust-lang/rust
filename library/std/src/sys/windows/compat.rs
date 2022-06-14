//! A "compatibility layer" for supporting older versions of Windows
//!
//! The standard library uses some Windows API functions that are not present
//! on older versions of Windows.  (Note that the oldest version of Windows
//! that Rust supports is Windows 7 (client) and Windows Server 2008 (server).)
//! This module implements a form of delayed DLL import binding, using
//! `GetModuleHandle` and `GetProcAddress` to look up DLL entry points at
//! runtime.
//!
//! This implementation uses a static initializer to look up the DLL entry
//! points. The CRT (C runtime) executes static initializers before `main`
//! is called (for binaries) and before `DllMain` is called (for DLLs).
//! This is the ideal time to look up DLL imports, because we are guaranteed
//! that no other threads will attempt to call these entry points. Thus,
//! we can look up the imports and store them in `static mut` fields
//! without any synchronization.
//!
//! This has an additional advantage: Because the DLL import lookup happens
//! at module initialization, the cost of these lookups is deterministic,
//! and is removed from the code paths that actually call the DLL imports.
//! That is, there is no unpredictable "cache miss" that occurs when calling
//! a DLL import. For applications that benefit from predictable delays,
//! this is a benefit. This also eliminates the comparison-and-branch
//! from the hot path.
//!
//! Currently, the standard library uses only a small number of dynamic
//! DLL imports. If this number grows substantially, then the cost of
//! performing all of the lookups at initialization time might become
//! substantial.
//!
//! The mechanism of registering a static initializer with the CRT is
//! documented in
//! [CRT Initialization](https://docs.microsoft.com/en-us/cpp/c-runtime-library/crt-initialization?view=msvc-160).
//! It works by contributing a global symbol to the `.CRT$XCU` section.
//! The linker builds a table of all static initializer functions.
//! The CRT startup code then iterates that table, calling each
//! initializer function.
//!
//! # **WARNING!!*
//! The environment that a static initializer function runs in is highly
//! constrained. There are **many** restrictions on what static initializers
//! can safely do. Static initializer functions **MUST NOT** do any of the
//! following (this list is not comprehensive):
//! * touch any other static field that is used by a different static
//!   initializer, because the order that static initializers run in
//!   is not defined.
//! * call `LoadLibrary` or any other function that acquires the DLL
//!   loader lock.
//! * call any Rust function or CRT function that touches any static
//!   (global) state.

macro_rules! compat_fn {
    ($module:literal: $(
        $(#[$meta:meta])*
        pub fn $symbol:ident($($argname:ident: $argtype:ty),*) -> $rettype:ty $fallback_body:block
    )*) => ($(
        $(#[$meta])*
        pub mod $symbol {
            #[allow(unused_imports)]
            use super::*;
            use crate::mem;

            type F = unsafe extern "system" fn($($argtype),*) -> $rettype;

            /// Points to the DLL import, or the fallback function.
            ///
            /// This static can be an ordinary, unsynchronized, mutable static because
            /// we guarantee that all of the writes finish during CRT initialization,
            /// and all of the reads occur after CRT initialization.
            static mut PTR: Option<F> = None;

            /// This symbol is what allows the CRT to find the `init` function and call it.
            /// It is marked `#[used]` because otherwise Rust would assume that it was not
            /// used, and would remove it.
            #[used]
            #[link_section = ".CRT$XCU"]
            static INIT_TABLE_ENTRY: unsafe extern "C" fn() = init;

            unsafe extern "C" fn init() {
                PTR = get_f();
            }

            unsafe extern "C" fn get_f() -> Option<F> {
                // There is no locking here. This code is executed before main() is entered, and
                // is guaranteed to be single-threaded.
                //
                // DO NOT do anything interesting or complicated in this function! DO NOT call
                // any Rust functions or CRT functions, if those functions touch any global state,
                // because this function runs during global initialization. For example, DO NOT
                // do any dynamic allocation, don't call LoadLibrary, etc.
                let module_name: *const u8 = concat!($module, "\0").as_ptr();
                let symbol_name: *const u8 = concat!(stringify!($symbol), "\0").as_ptr();
                let module_handle = $crate::sys::c::GetModuleHandleA(module_name as *const i8);
                if !module_handle.is_null() {
                    let ptr = $crate::sys::c::GetProcAddress(module_handle, symbol_name as *const i8);
                    if !ptr.is_null() {
                        // Transmute to the right function pointer type.
                        return Some(mem::transmute(ptr));
                    }
                }
                return None;
            }

            #[allow(dead_code)]
            #[inline(always)]
            pub fn option() -> Option<F> {
                unsafe {
                    if cfg!(miri) {
                        // Miri does not run `init`, so we just call `get_f` each time.
                        get_f()
                    } else {
                        PTR
                    }
                }
            }

            #[allow(dead_code)]
            pub unsafe fn call($($argname: $argtype),*) -> $rettype {
                if let Some(ptr) = option() {
                    return ptr($($argname),*);
                }
                $fallback_body
            }
        }

        $(#[$meta])*
        pub use $symbol::call as $symbol;
    )*)
}
