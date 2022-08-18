//! A "compatibility layer" for supporting older versions of Windows
//!
//! The standard library uses some Windows API functions that are not present
//! on older versions of Windows.  (Note that the oldest version of Windows
//! that Rust supports is Windows 7 (client) and Windows Server 2008 (server).)
//! This module implements a form of delayed DLL import binding, using
//! `GetModuleHandle` and `GetProcAddress` to look up DLL entry points at
//! runtime.
//!
//! This is implemented simply by storing a function pointer in an atomic.
//! Loading and calling this function will have little or no overhead
//! compared with calling any other dynamically imported function.
//!
//! The stored function pointer starts out as an importer function which will
//! swap itself with the real function when it's called for the first time. If
//! the real function can't be imported then a fallback function is used in its
//! place. While this is low cost for the happy path (where the function is
//! already loaded) it does mean there's some overhead the first time the
//! function is called. In the worst case, multiple threads may all end up
//! importing the same function unnecessarily.

use crate::ffi::{c_void, CStr};
use crate::ptr::NonNull;
use crate::sync::atomic::{AtomicBool, Ordering};
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

    /// Load the library (if not already loaded)
    ///
    /// # Safety
    ///
    /// The module must not be unloaded.
    pub unsafe fn load_system_library(name: &CStr) -> Option<Self> {
        let module = c::LoadLibraryExA(
            name.as_ptr(),
            crate::ptr::null_mut(),
            c::LOAD_LIBRARY_SEARCH_SYSTEM32,
        );
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
        $vis:vis fn $symbol:ident($($argname:ident: $argtype:ty),*) -> $rettype:ty $fallback_body:block
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
                    static SYMBOL_NAME: &CStr = ansi_str!(sym $symbol);
                    if let Some(f) = module.and_then(|m| m.proc_address(SYMBOL_NAME)) {
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

            #[inline(always)]
            pub unsafe fn call($($argname: $argtype),*) -> $rettype {
                let func: F = mem::transmute(PTR.load(Ordering::Relaxed));
                func($($argname),*)
            }
        }
        $(#[$meta])*
        $vis use $symbol::call as $symbol;
    )*)
}

/// Optionally loaded functions.
///
/// Actual loading of the function defers to $load_functions.
macro_rules! compat_fn_optional {
    ($load_functions:expr;
    $(
        $(#[$meta:meta])*
        $vis:vis fn $symbol:ident($($argname:ident: $argtype:ty),*) $(-> $rettype:ty)?;
    )+) => (
        $(
            pub mod $symbol {
                use super::*;
                use crate::ffi::c_void;
                use crate::mem;
                use crate::ptr::{self, NonNull};
                use crate::sync::atomic::{AtomicPtr, Ordering};

                pub(in crate::sys) static PTR: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());

                type F = unsafe extern "system" fn($($argtype),*) $(-> $rettype)?;

                #[inline(always)]
                pub fn option() -> Option<F> {
                    let f = PTR.load(Ordering::Acquire);
                    if !f.is_null() { Some(unsafe { mem::transmute(f) }) } else { try_load() }
                }

                #[cold]
                fn try_load() -> Option<F> {
                    $load_functions;
                    NonNull::new(PTR.load(Ordering::Acquire)).map(|f| unsafe { mem::transmute(f) })
                }
            }
        )+
    )
}

/// Load all needed functions from "api-ms-win-core-synch-l1-2-0".
pub(super) fn load_synch_functions() {
    fn try_load() -> Option<()> {
        const MODULE_NAME: &CStr = ansi_str!("api-ms-win-core-synch-l1-2-0");
        const WAIT_ON_ADDRESS: &CStr = ansi_str!("WaitOnAddress");
        const WAKE_BY_ADDRESS_SINGLE: &CStr = ansi_str!("WakeByAddressSingle");

        // Try loading the library and all the required functions.
        // If any step fails, then they all fail.
        let library = unsafe { Module::load_system_library(MODULE_NAME) }?;
        let wait_on_address = library.proc_address(WAIT_ON_ADDRESS)?;
        let wake_by_address_single = library.proc_address(WAKE_BY_ADDRESS_SINGLE)?;

        c::WaitOnAddress::PTR.store(wait_on_address.as_ptr(), Ordering::Release);
        c::WakeByAddressSingle::PTR.store(wake_by_address_single.as_ptr(), Ordering::Release);
        Some(())
    }

    // Try to load the module but skip loading if a previous attempt failed.
    static LOAD_MODULE: AtomicBool = AtomicBool::new(true);
    let module_loaded = LOAD_MODULE.load(Ordering::Acquire) && try_load().is_some();
    LOAD_MODULE.store(module_loaded, Ordering::Release)
}
