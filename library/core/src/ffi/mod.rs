//! Platform-specific types, as defined by C.
//!
//! Code that interacts via FFI will almost certainly be using the
//! base types provided by C, which aren't nearly as nicely defined
//! as Rust's primitive types. This module provides types which will
//! match those defined by C, so that code that interacts with C will
//! refer to the correct types.

#![stable(feature = "core_ffi", since = "1.30.0")]
#![allow(non_camel_case_types)]

use crate::fmt;
use crate::panic::RefUnwindSafe;
#[stable(feature = "c_str_module", since = "1.88.0")]
pub mod c_str;
#[doc(inline)]
#[stable(feature = "core_c_str", since = "1.64.0")]
pub use self::c_str::CStr;
#[doc(inline)]
#[stable(feature = "cstr_from_bytes_until_nul", since = "1.69.0")]
pub use self::c_str::FromBytesUntilNulError;
#[doc(inline)]
#[stable(feature = "core_c_str", since = "1.64.0")]
pub use self::c_str::FromBytesWithNulError;

mod primitives;
#[stable(feature = "core_ffi_c", since = "1.64.0")]
pub use self::primitives::{
    c_char, c_double, c_float, c_int, c_long, c_longlong, c_schar, c_short, c_uchar, c_uint,
    c_ulong, c_ulonglong, c_ushort,
};
#[unstable(feature = "c_size_t", issue = "88345")]
pub use self::primitives::{c_ptrdiff_t, c_size_t, c_ssize_t};

mod va_list;
#[stable(feature = "c_variadic", since = "CURRENT_RUSTC_VERSION")]
pub use self::va_list::{VaArgSafe, VaList};

#[doc = include_str!("c_void.md")]
#[lang = "c_void"]
#[repr(transparent)]
#[stable(feature = "core_c_void", since = "1.30.0")]
pub struct c_void {
    // Using this weird type ensures a size of 1,
    // while minimizing UB if a user incorrectly tries
    // to dereference a pointer to `c_void`,
    // or reborrow it as a reference.
    #[cfg(not(miri))]
    _inner: crate::pin::UnsafePinned<crate::mem::MaybeUninit<u8>>,

    // However, if running in Miri,
    // we want to maximize detection of UB,
    // so we make `c_void` uninhabited.
    #[cfg(miri)]
    _inner: u8,
    #[cfg(miri)]
    _uninhabited: !,
}

// for backward compatibility.
#[stable(feature = "core_c_void", since = "1.30.0")]
impl RefUnwindSafe for c_void {}

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for c_void {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("c_void").finish()
    }
}

// Link the MSVC default lib
#[cfg(all(windows, target_env = "msvc"))]
#[link(
    name = "/defaultlib:msvcrt",
    modifiers = "+verbatim",
    cfg(not(target_feature = "crt-static"))
)]
#[link(name = "/defaultlib:libcmt", modifiers = "+verbatim", cfg(target_feature = "crt-static"))]
unsafe extern "C" {}

// Used by rustc for checking the definitions of other function with the same symbol names
//
// See the `invalid_runtime_symbols_definitions` lint.
mod runtime_symbols {
    use crate::ffi::{c_char, c_int, c_void};

    unsafe extern "C" {
        #[rustc_canonical_symbol]
        fn memcpy(dest: *mut c_void, src: *const c_void, n: usize) -> *mut c_void;

        #[rustc_canonical_symbol]
        fn memmove(dest: *mut c_void, src: *const c_void, n: usize) -> *mut c_void;

        #[rustc_canonical_symbol]
        fn memset(s: *mut c_void, c: c_int, n: usize) -> *mut c_void;

        #[rustc_canonical_symbol]
        fn memcmp(s1: *const c_void, s2: *const c_void, n: usize) -> c_int;

        #[rustc_canonical_symbol]
        fn bcmp(s1: *const c_void, s2: *const c_void, n: usize) -> c_int;

        #[rustc_canonical_symbol]
        fn strlen(s: *const c_char) -> usize;
    }
}
