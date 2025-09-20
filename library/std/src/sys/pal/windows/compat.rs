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

use crate::ffi::{CStr, c_void};
use crate::ptr::NonNull;
use crate::sys::c;

/// Helper macro for creating CStrs from literals and symbol names.
macro_rules! ansi_str {
    (sym $ident:ident) => {{ crate::sys::compat::const_cstr_from_bytes(concat!(stringify!($ident), "\0").as_bytes()) }};
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
        unsafe {
            let module = c::GetModuleHandleA(name.as_ptr().cast::<u8>());
            NonNull::new(module).map(Self)
        }
    }

    // Try to get the address of a function.
    pub fn proc_address(self, name: &CStr) -> Option<NonNull<c_void>> {
        unsafe {
            // SAFETY:
            // `self.0` will always be a valid module.
            // A CStr is always null terminated.
            let proc = c::GetProcAddress(self.0.as_ptr(), name.as_ptr().cast::<u8>());
            // SAFETY: `GetProcAddress` returns None on null.
            proc.map(|p| NonNull::new_unchecked(p as *mut c_void))
        }
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
            use crate::sync::atomic::{Atomic, AtomicPtr, Ordering};
            use crate::sys::compat::Module;

            type F = unsafe extern "system" fn($($argtype),*) -> $rettype;

            /// `PTR` contains a function pointer to one of three functions.
            /// It starts with the `load` function.
            /// When that is called it attempts to load the requested symbol.
            /// If it succeeds, `PTR` is set to the address of that symbol.
            /// If it fails, then `PTR` is set to `fallback`.
            static PTR: Atomic<*mut c_void> = AtomicPtr::new(load as *mut _);

            unsafe extern "system" fn load($($argname: $argtype),*) -> $rettype {
                unsafe {
                    let func = load_from_module(Module::new($module));
                    func($($argname),*)
                }
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
                unsafe {
                    let func: F = mem::transmute(PTR.load(Ordering::Relaxed));
                    func($($argname),*)
                }
            }
        }
        #[allow(unused)]
        $(#[$meta])*
        $vis use $symbol::call as $symbol;
    )*)
}

/// Optionally loaded functions.
///
/// Relies on the functions being pre-loaded elsewhere.
#[cfg(target_vendor = "win7")]
macro_rules! compat_fn_optional {
    (pub static $module:ident: &CStr = $name:expr; $(
        $(#[$meta:meta])*
        $vis:vis fn $symbol:ident($($argname:ident: $argtype:ty),*) $(-> $rettype:ty)?;
    )+) => (
        pub static $module: &CStr = $name;
        $(
            pub mod $symbol {
                #[allow(unused_imports)]
                use super::*;
                use crate::ffi::c_void;
                use crate::mem;
                use crate::ptr;
                use crate::sync::atomic::{Atomic, AtomicPtr, Ordering};
                use crate::sys::compat::Module;

                const NOT_FOUND: *mut c_void = ptr::null_mut();
                const NOT_LOADED: *mut c_void = ptr::without_provenance_mut(usize::MAX);

                pub(in crate::sys) static PTR: Atomic<*mut c_void> = AtomicPtr::new(NOT_LOADED);

                type F = unsafe extern "system" fn($($argtype),*) $(-> $rettype)?;

                fn load_from_module(module: Option<Module>) -> Option<F> {
                    unsafe {
                        static SYMBOL_NAME: &CStr = ansi_str!(sym $symbol);
                        if let Some(f) = module.and_then(|m| m.proc_address(SYMBOL_NAME)) {
                            PTR.store(f.as_ptr(), Ordering::Relaxed);
                            Some(mem::transmute(f))
                        } else {
                            PTR.store(NOT_FOUND, Ordering::Relaxed);
                            None
                        }
                    }
                }

                pub fn option() -> Option<F> {
                    match PTR.load(Ordering::Relaxed) {
                        NOT_FOUND => None,
                        NOT_LOADED => load_from_module(unsafe { Module::new($module) }),
                        f => Some(unsafe { mem::transmute(f) })
                    }
                }
            }
            #[inline]
            pub unsafe extern "system" fn $symbol($($argname: $argtype),*) $(-> $rettype)? {
                unsafe { $symbol::option().unwrap()($($argname),*) }
            }
        )+
    )
}
