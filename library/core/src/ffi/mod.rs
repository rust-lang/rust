//! Platform-specific types, as defined by C.
//!
//! Code that interacts via FFI will almost certainly be using the
//! base types provided by C, which aren't nearly as nicely defined
//! as Rust's primitive types. This module provides types which will
//! match those defined by C, so that code that interacts with C will
//! refer to the correct types.

#![stable(feature = "core_ffi", since = "1.30.0")]
#![allow(non_camel_case_types)]

#[doc(inline)]
#[stable(feature = "core_c_str", since = "1.64.0")]
pub use self::c_str::CStr;
#[doc(inline)]
#[stable(feature = "cstr_from_bytes_until_nul", since = "1.69.0")]
pub use self::c_str::FromBytesUntilNulError;
#[doc(inline)]
#[stable(feature = "core_c_str", since = "1.64.0")]
pub use self::c_str::FromBytesWithNulError;
use crate::fmt;

#[unstable(feature = "c_str_module", issue = "112134")]
pub mod c_str;

#[unstable(
    feature = "c_variadic",
    issue = "44930",
    reason = "the `c_variadic` feature has not been properly tested on all supported platforms"
)]
pub use self::va_list::{VaList, VaListImpl};

#[unstable(
    feature = "c_variadic",
    issue = "44930",
    reason = "the `c_variadic` feature has not been properly tested on all supported platforms"
)]
pub mod va_list;

mod primitives;
#[stable(feature = "core_ffi_c", since = "1.64.0")]
pub use self::primitives::{
    c_char, c_double, c_float, c_int, c_long, c_longlong, c_schar, c_short, c_uchar, c_uint,
    c_ulong, c_ulonglong, c_ushort,
};
#[unstable(feature = "c_size_t", issue = "88345")]
pub use self::primitives::{c_ptrdiff_t, c_size_t, c_ssize_t};

// N.B., for LLVM to recognize the void pointer type and by extension
//     functions like malloc(), we need to have it represented as i8* in
//     LLVM bitcode. The enum used here ensures this and prevents misuse
//     of the "raw" type by only having private variants. We need two
//     variants, because the compiler complains about the repr attribute
//     otherwise and we need at least one variant as otherwise the enum
//     would be uninhabited and at least dereferencing such pointers would
//     be UB.
#[doc = include_str!("c_void.md")]
#[lang = "c_void"]
#[cfg_attr(not(doc), repr(u8))] // An implementation detail we don't want to show up in rustdoc
#[stable(feature = "core_c_void", since = "1.30.0")]
pub enum c_void {
    #[unstable(
        feature = "c_void_variant",
        reason = "temporary implementation detail",
        issue = "none"
    )]
    #[doc(hidden)]
    __variant1,
    #[unstable(
        feature = "c_void_variant",
        reason = "temporary implementation detail",
        issue = "none"
    )]
    #[doc(hidden)]
    __variant2,
}

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
extern "C" {}
