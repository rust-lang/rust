//! A "compatibility layer" for supporting older versions of Windows
//!
//! The standard library uses some Windows API functions that are not present
//! on older versions of Windows.  (Note that the oldest version of Windows
//! that Rust supports is Windows 7 (client) and Windows Server 2008 (server).)
//! This module implements a form of delayed DLL import binding, using
//! `GetModuleHandle` and `GetProcAddress` to look up DLL entry points at
//! runtime.
//!
//! This is implemented simply by storing a function pointer in an atomic
//! and using relaxed ordering to load it. This means that calling it will be no
//! more expensive then calling any other dynamically imported function.
//!
//! The stored function pointer starts out as an importer function which will
//! swap itself with the real function when it's called for the first time. If
//! the real function can't be imported then a fallback function is used in its
//! place. While this is zero cost for the happy path (where the function is
//! already loaded) it does mean there's some overhead the first time the
//! function is called. In the worst case, multiple threads may all end up
//! importing the same function unnecessarily.

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

/// Optionally load `WaitOnAddress`.
/// Unlike the dynamic loading described above, this does not have a fallback.
///
/// This is rexported from sys::c. You should prefer to import
/// from there in case this changes again in the future.
pub mod WaitOnAddress {
    use super::*;
    use crate::mem;
    use crate::ptr;
    use crate::sync::atomic::{AtomicBool, AtomicPtr, Ordering};
    use crate::sys::c;

    static MODULE_NAME: &CStr = ansi_str!("api-ms-win-core-synch-l1-2-0");
    static SYMBOL_NAME: &CStr = ansi_str!("WaitOnAddress");

    // WaitOnAddress function signature.
    type F = unsafe extern "system" fn(
        Address: c::LPVOID,
        CompareAddress: c::LPVOID,
        AddressSize: c::SIZE_T,
        dwMilliseconds: c::DWORD,
    );

    // A place to store the loaded function atomically.
    static WAIT_ON_ADDRESS: AtomicPtr<c_void> = AtomicPtr::new(ptr::null_mut());

    // We can skip trying to load again if we already tried.
    static LOAD_MODULE: AtomicBool = AtomicBool::new(true);

    #[inline(always)]
    pub fn option() -> Option<F> {
        let f = WAIT_ON_ADDRESS.load(Ordering::Relaxed);
        if !f.is_null() { Some(unsafe { mem::transmute(f) }) } else { try_load() }
    }

    #[cold]
    fn try_load() -> Option<F> {
        if LOAD_MODULE.load(Ordering::Acquire) {
            // load the module
            let mut wait_on_address = None;
            if let Some(func) = try_load_inner() {
                WAIT_ON_ADDRESS.store(func.as_ptr(), Ordering::Relaxed);
                wait_on_address = Some(unsafe { mem::transmute(func) });
            }
            // Don't try to load the module again even if loading failed.
            LOAD_MODULE.store(false, Ordering::Release);
            wait_on_address
        } else {
            None
        }
    }

    // In the future this could be a `try` block but until then I think it's a
    // little bit cleaner as a separate function.
    fn try_load_inner() -> Option<NonNull<c_void>> {
        unsafe { Module::new(MODULE_NAME)?.proc_address(SYMBOL_NAME) }
    }
}
