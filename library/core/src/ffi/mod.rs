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

#[stable(feature = "c_str_module", since = "1.88.0")]
pub mod c_str;

#[unstable(
    feature = "c_variadic",
    issue = "44930",
    reason = "the `c_variadic` feature has not been properly tested on all supported platforms"
)]
pub use self::va_list::{VaArgSafe, VaList};

#[unstable(
    feature = "c_variadic",
    issue = "44930",
    reason = "the `c_variadic` feature has not been properly tested on all supported platforms"
)]
pub mod va_list;

mod primitives;

macro_rules! type_alias {
    {
      $Docfile:tt, $Alias:ident = $Real:ty;
      $( $Cfg:tt )*
    } => {
        #[doc = include_str!($Docfile)]
        #[doc(cfg(all()))]
        $( $Cfg )*
        pub type $Alias = $Real;
    }
}

type_alias! { "c_char.md", c_char = self::primitives::c_char; #[stable(feature = "core_ffi_c", since = "1.64.0")]}

type_alias! { "c_schar.md", c_schar = self::primitives::c_schar; #[stable(feature = "core_ffi_c", since = "1.64.0")]}
type_alias! { "c_uchar.md", c_uchar = self::primitives::c_uchar; #[stable(feature = "core_ffi_c", since = "1.64.0")]}

type_alias! { "c_short.md", c_short = self::primitives::c_short; #[stable(feature = "core_ffi_c", since = "1.64.0")]}
type_alias! { "c_ushort.md", c_ushort = self::primitives::c_ushort; #[stable(feature = "core_ffi_c", since = "1.64.0")]}

type_alias! { "c_int.md", c_int = self::primitives::c_int; #[stable(feature = "core_ffi_c", since = "1.64.0")]}
type_alias! { "c_uint.md", c_uint = self::primitives::c_uint; #[stable(feature = "core_ffi_c", since = "1.64.0")]}

type_alias! { "c_long.md", c_long = self::primitives::c_long; #[stable(feature = "core_ffi_c", since = "1.64.0")]}
type_alias! { "c_ulong.md", c_ulong = self::primitives::c_ulong; #[stable(feature = "core_ffi_c", since = "1.64.0")]}

type_alias! { "c_longlong.md", c_longlong = self::primitives::c_longlong; #[stable(feature = "core_ffi_c", since = "1.64.0")]}
type_alias! { "c_ulonglong.md", c_ulonglong = self::primitives::c_ulonglong; #[stable(feature = "core_ffi_c", since = "1.64.0")]}

type_alias! { "c_float.md", c_float = self::primitives::c_float; #[stable(feature = "core_ffi_c", since = "1.64.0")]}
type_alias! { "c_double.md", c_double = self::primitives::c_double; #[stable(feature = "core_ffi_c", since = "1.64.0")]}

type_alias! { "c_size_t.md", c_size_t = self::primitives::c_size_t; #[unstable(feature = "c_size_t", issue = "88345")]}
type_alias! { "c_ptrdiff_t.md", c_ptrdiff_t = self::primitives::c_ptrdiff_t; #[unstable(feature = "c_size_t", issue = "88345")]}
type_alias! { "c_ssize_t.md", c_ssize_t = self::primitives::c_ssize_t; #[unstable(feature = "c_size_t", issue = "88345")]}

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
#[repr(u8)]
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
unsafe extern "C" {}
