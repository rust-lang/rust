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
#[doc(no_inline)]
#[stable(feature = "cstr_from_bytes_until_nul", since = "1.69.0")]
pub use self::c_str::FromBytesUntilNulError;
#[doc(no_inline)]
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

macro_rules! type_alias {
    {
      $Docfile:tt, $Alias:ident = $Real:ty;
      $( $Cfg:tt )*
    } => {
        #[doc = include_str!($Docfile)]
        $( $Cfg )*
        #[stable(feature = "core_ffi_c", since = "1.64.0")]
        pub type $Alias = $Real;
    }
}

type_alias! { "c_char.md", c_char = c_char_definition::c_char; #[doc(cfg(all()))] }

type_alias! { "c_schar.md", c_schar = i8; }
type_alias! { "c_uchar.md", c_uchar = u8; }
type_alias! { "c_short.md", c_short = i16; }
type_alias! { "c_ushort.md", c_ushort = u16; }

type_alias! { "c_int.md", c_int = c_int_definition::c_int; #[doc(cfg(all()))] }
type_alias! { "c_uint.md", c_uint = c_int_definition::c_uint; #[doc(cfg(all()))] }

type_alias! { "c_long.md", c_long = c_long_definition::c_long; #[doc(cfg(all()))] }
type_alias! { "c_ulong.md", c_ulong = c_long_definition::c_ulong; #[doc(cfg(all()))] }

type_alias! { "c_longlong.md", c_longlong = i64; }
type_alias! { "c_ulonglong.md", c_ulonglong = u64; }

type_alias! { "c_float.md", c_float = f32; }
type_alias! { "c_double.md", c_double = f64; }

/// Equivalent to C's `size_t` type, from `stddef.h` (or `cstddef` for C++).
///
/// This type is currently always [`usize`], however in the future there may be
/// platforms where this is not the case.
#[unstable(feature = "c_size_t", issue = "88345")]
pub type c_size_t = usize;

/// Equivalent to C's `ptrdiff_t` type, from `stddef.h` (or `cstddef` for C++).
///
/// This type is currently always [`isize`], however in the future there may be
/// platforms where this is not the case.
#[unstable(feature = "c_size_t", issue = "88345")]
pub type c_ptrdiff_t = isize;

/// Equivalent to C's `ssize_t` (on POSIX) or `SSIZE_T` (on Windows) type.
///
/// This type is currently always [`isize`], however in the future there may be
/// platforms where this is not the case.
#[unstable(feature = "c_size_t", issue = "88345")]
pub type c_ssize_t = isize;

mod c_char_definition {
    cfg_if! {
        // These are the targets on which c_char is unsigned. Usually the
        // signedness is the same for all target_os values on a given
        // architecture but there are some exceptions (see isSignedCharDefault()
        // in clang/lib/Driver/ToolChains/Clang.cpp):
        // - PowerPC uses unsigned char for all targets except Darwin
        // - Arm/AArch64 uses unsigned char except for Darwin and Windows
        // - L4RE builds with -funsigned-char on all targets
        if #[cfg(all(
            not(windows),
            not(target_vendor = "apple"),
            any(
                target_arch = "aarch64",
                target_arch = "arm",
                target_arch = "csky",
                target_arch = "hexagon",
                target_arch = "msp430",
                target_arch = "powerpc",
                target_arch = "powerpc64",
                target_arch = "riscv64",
                target_arch = "riscv32",
                target_arch = "s390x",
                target_arch = "xtensa",
                target_os = "l4re",
            )
        ))] {
            pub type c_char = u8;
        } else {
            // On every other target, c_char is signed.
            pub type c_char = i8;
        }
    }
}

mod c_int_definition {
    cfg_if! {
        if #[cfg(any(target_arch = "avr", target_arch = "msp430"))] {
            pub type c_int = i16;
            pub type c_uint = u16;
        } else {
            pub type c_int = i32;
            pub type c_uint = u32;
        }
    }
}

mod c_long_definition {
    cfg_if! {
        if #[cfg(all(target_pointer_width = "64", not(windows)))] {
            pub type c_long = i64;
            pub type c_ulong = u64;
        } else {
            // The minimal size of `long` in the C standard is 32 bits
            pub type c_long = i32;
            pub type c_ulong = u32;
        }
    }
}

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
