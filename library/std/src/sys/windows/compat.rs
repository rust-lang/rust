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

use crate::ffi::{c_void, CStr};
use crate::ptr::NonNull;
use crate::sys::c;

/// Helper macro for creating CStrs from literals and symbol names.
macro_rules! ansi_str {
    (sym $ident:ident) => {{
        #[allow(unused_unsafe)]
        crate::sys::compat::const_cstr_from_bytes(concat!(stringify!($ident), "\0").as_bytes())
    }};
    ($lit:literal) => {{ crate::sys::compat::const_cstr_from_bytes(concat!($lit, "\0").as_bytes()) }};
}

/// Creates a C string wrapper from a byte slice, in a constant context.
///
/// This is a utility function used by the [`ansi_str`] macro.
///
/// # Panics
///
/// Panics if the slice is not null terminated or contains nulls, except as the last item
pub(crate) const fn const_cstr_from_bytes(bytes: &'static [u8]) -> &'static CStr {
    if !matches!(bytes.last(), Some(&0)) {
        panic!("A CStr must be null terminated");
    }
    let mut i = 0;
    // At this point `len()` is at least 1.
    while i < bytes.len() - 1 {
        if bytes[i] == 0 {
            panic!("A CStr must not have interior nulls")
        }
        i += 1;
    }
    // SAFETY: The safety is ensured by the above checks.
    unsafe { crate::ffi::CStr::from_bytes_with_nul_unchecked(bytes) }
}

#[used]
#[link_section = ".CRT$XCU"]
static INIT_TABLE_ENTRY: unsafe extern "C" fn() = init;

/// This is where the magic preloading of symbols happens.
///
/// Note that any functions included here will be unconditionally included in
/// the final binary, regardless of whether or not they're actually used.
///
/// Therefore, this is limited to `compat_fn_optional` functions which must be
/// preloaded and any functions which may be more time sensitive, even for the first call.
unsafe extern "C" fn init() {
    // There is no locking here. This code is executed before main() is entered, and
    // is guaranteed to be single-threaded.
    //
    // DO NOT do anything interesting or complicated in this function! DO NOT call
    // any Rust functions or CRT functions if those functions touch any global state,
    // because this function runs during global initialization. For example, DO NOT
    // do any dynamic allocation, don't call LoadLibrary, etc.

    if let Some(synch) = Module::new(c::SYNCH_API) {
        // These are optional and so we must manually attempt to load them
        // before they can be used.
        c::WaitOnAddress::preload(synch);
        c::WakeByAddressSingle::preload(synch);
    }

    if let Some(kernel32) = Module::new(c::KERNEL32) {
        // Preloading this means getting a precise time will be as fast as possible.
        c::GetSystemTimePreciseAsFileTime::preload(kernel32);
    }
}

/// Represents a loaded module.
///
/// Note that the modules std depends on must not be unloaded.
/// Therefore a `Module` is always valid for the lifetime of std.
#[derive(Copy, Clone)]
pub(in crate::sys) struct Module(NonNull<c_void>);
impl Module {
    /// Try to get a handle to a loaded module.
    ///
    /// # SAFETY
    ///
    /// This should only be use for modules that exist for the lifetime of std
    /// (e.g. kernel32 and ntdll).
    pub unsafe fn new(name: &CStr) -> Option<Self> {
        // SAFETY: A CStr is always null terminated.
        let module = c::GetModuleHandleA(name.as_ptr());
        NonNull::new(module).map(Self)
    }

    // Try to get the address of a function.
    pub fn proc_address(self, name: &CStr) -> Option<NonNull<c_void>> {
        // SAFETY:
        // `self.0` will always be a valid module.
        // A CStr is always null terminated.
        let proc = unsafe { c::GetProcAddress(self.0.as_ptr(), name.as_ptr()) };
        NonNull::new(proc)
    }
}

/// Load a function or use a fallback implementation if that fails.
macro_rules! compat_fn_with_fallback {
    (pub static $module:ident: &CStr = $name:expr; $(
        $(#[$meta:meta])*
        pub fn $symbol:ident($($argname:ident: $argtype:ty),*) -> $rettype:ty $fallback_body:block
    )*) => (
        pub static $module: &CStr = $name;
    $(
        $(#[$meta])*
        pub mod $symbol {
            #[allow(unused_imports)]
            use super::*;
            use crate::mem;
            use crate::ffi::CStr;
            use crate::sync::atomic::{AtomicPtr, Ordering};
            use crate::sys::compat::Module;

            type F = unsafe extern "system" fn($($argtype),*) -> $rettype;

            /// `PTR` contains a function pointer to one of three functions.
            /// It starts with the `load` function.
            /// When that is called it attempts to load the requested symbol.
            /// If it succeeds, `PTR` is set to the address of that symbol.
            /// If it fails, then `PTR` is set to `fallback`.
            static PTR: AtomicPtr<c_void> = AtomicPtr::new(load as *mut _);

            unsafe extern "system" fn load($($argname: $argtype),*) -> $rettype {
                let func = load_from_module(Module::new($module));
                func($($argname),*)
            }

            fn load_from_module(module: Option<Module>) -> F {
                unsafe {
                    static symbol_name: &CStr = ansi_str!(sym $symbol);
                    if let Some(f) = module.and_then(|m| m.proc_address(symbol_name)) {
                        PTR.store(f.as_ptr(), Ordering::Relaxed);
                        mem::transmute(f)
                    } else {
                        PTR.store(fallback as *mut _, Ordering::Relaxed);
                        fallback
                    }
                }
            }

            #[allow(unused_variables)]
            unsafe extern "system" fn fallback($($argname: $argtype),*) -> $rettype {
                $fallback_body
            }

            #[allow(unused)]
            pub(in crate::sys) fn preload(module: Module) {
                load_from_module(Some(module));
            }

            #[inline(always)]
            pub unsafe fn call($($argname: $argtype),*) -> $rettype {
                let func: F = mem::transmute(PTR.load(Ordering::Relaxed));
                func($($argname),*)
            }
        }
        $(#[$meta])*
        pub use $symbol::call as $symbol;
    )*)
}

/// A function that either exists or doesn't.
///
/// NOTE: Optional functions must be preloaded in the `init` function above, or they will always be None.
macro_rules! compat_fn_optional {
    (pub static $module:ident: &CStr = $name:expr; $(
        $(#[$meta:meta])*
        pub fn $symbol:ident($($argname:ident: $argtype:ty),*) -> $rettype:ty;
    )*) => (
        pub static $module: &CStr = $name;
    $(
        $(#[$meta])*
        pub mod $symbol {
            #[allow(unused_imports)]
            use super::*;
            use crate::mem;
            use crate::sync::atomic::{AtomicPtr, Ordering};
            use crate::sys::compat::Module;
            use crate::ptr::{self, NonNull};

            type F = unsafe extern "system" fn($($argtype),*) -> $rettype;

            /// `PTR` will either be `null()` or set to the loaded function.
            static PTR: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());

            /// Only allow access to the function if it has loaded successfully.
            #[inline(always)]
            #[cfg(not(miri))]
            pub fn option() -> Option<F> {
                unsafe {
                    NonNull::new(PTR.load(Ordering::Relaxed)).map(|f| mem::transmute(f))
                }
            }

            // Miri does not understand the way we do preloading
            // therefore load the function here instead.
            #[cfg(miri)]
            pub fn option() -> Option<F> {
                let mut func = NonNull::new(PTR.load(Ordering::Relaxed));
                if func.is_none() {
                    Module::new($module).map(preload);
                    func = NonNull::new(PTR.load(Ordering::Relaxed));
                }
                unsafe {
                    func.map(|f| mem::transmute(f))
                }
            }

            #[allow(unused)]
            pub(in crate::sys) fn preload(module: Module) {
                unsafe {
                    let symbol_name = ansi_str!(sym $symbol);
                    if let Some(f) = module.proc_address(symbol_name) {
                        PTR.store(f.as_ptr(), Ordering::Relaxed);
                    }
                }
            }
        }
    )*)
}
