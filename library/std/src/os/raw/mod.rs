//! Platform-specific types, as defined by C.
//!
//! Code that interacts via FFI will almost certainly be using the
//! base types provided by C, which aren't nearly as nicely defined
//! as Rust's primitive types. This module provides types which will
//! match those defined by C, so that code that interacts with C will
//! refer to the correct types.

#![stable(feature = "raw_os", since = "1.1.0")]

#[cfg(test)]
mod tests;

use core::num::*;

macro_rules! type_alias_no_nz {
    {
      $Docfile:tt, $Alias:ident = $Real:ty;
      $( $Cfg:tt )*
    } => {
        #[doc = include_str!($Docfile)]
        $( $Cfg )*
        #[stable(feature = "raw_os", since = "1.1.0")]
        pub type $Alias = $Real;
    }
}

// To verify that the NonZero types in this file's macro invocations correspond
//
//  perl -n < library/std/src/os/raw/mod.rs -e 'next unless m/type_alias\!/; die "$_ ?" unless m/, (c_\w+) = (\w+), NonZero_(\w+) = NonZero(\w+)/; die "$_ ?" unless $3 eq $1 and $4 eq ucfirst $2'
//
// NB this does not check that the main c_* types are right.

macro_rules! type_alias {
    {
      $Docfile:tt, $Alias:ident = $Real:ty, $NZAlias:ident = $NZReal:ty;
      $( $Cfg:tt )*
    } => {
        type_alias_no_nz! { $Docfile, $Alias = $Real; $( $Cfg )* }

        #[doc = concat!("Type alias for `NonZero` version of [`", stringify!($Alias), "`]")]
        #[unstable(feature = "raw_os_nonzero", issue = "82363")]
        $( $Cfg )*
        pub type $NZAlias = $NZReal;
    }
}

type_alias! { "char.md", c_char = u8, NonZero_c_char = NonZeroU8;
#[cfg(any(
    all(
        target_os = "linux",
        any(
            target_arch = "aarch64",
            target_arch = "arm",
            target_arch = "hexagon",
            target_arch = "powerpc",
            target_arch = "powerpc64",
            target_arch = "s390x",
            target_arch = "riscv64",
            target_arch = "riscv32"
        )
    ),
    all(target_os = "android", any(target_arch = "aarch64", target_arch = "arm")),
    all(target_os = "l4re", target_arch = "x86_64"),
    all(
        target_os = "freebsd",
        any(
            target_arch = "aarch64",
            target_arch = "arm",
            target_arch = "powerpc",
            target_arch = "powerpc64"
        )
    ),
    all(
        target_os = "netbsd",
        any(target_arch = "aarch64", target_arch = "arm", target_arch = "powerpc")
    ),
    all(target_os = "openbsd", target_arch = "aarch64"),
    all(
        target_os = "vxworks",
        any(
            target_arch = "aarch64",
            target_arch = "arm",
            target_arch = "powerpc64",
            target_arch = "powerpc"
        )
    ),
    all(target_os = "fuchsia", target_arch = "aarch64")
))]}
type_alias! { "char.md", c_char = i8, NonZero_c_char = NonZeroI8;
#[cfg(not(any(
    all(
        target_os = "linux",
        any(
            target_arch = "aarch64",
            target_arch = "arm",
            target_arch = "hexagon",
            target_arch = "powerpc",
            target_arch = "powerpc64",
            target_arch = "s390x",
            target_arch = "riscv64",
            target_arch = "riscv32"
        )
    ),
    all(target_os = "android", any(target_arch = "aarch64", target_arch = "arm")),
    all(target_os = "l4re", target_arch = "x86_64"),
    all(
        target_os = "freebsd",
        any(
            target_arch = "aarch64",
            target_arch = "arm",
            target_arch = "powerpc",
            target_arch = "powerpc64"
        )
    ),
    all(
        target_os = "netbsd",
        any(target_arch = "aarch64", target_arch = "arm", target_arch = "powerpc")
    ),
    all(target_os = "openbsd", target_arch = "aarch64"),
    all(
        target_os = "vxworks",
        any(
            target_arch = "aarch64",
            target_arch = "arm",
            target_arch = "powerpc64",
            target_arch = "powerpc"
        )
    ),
    all(target_os = "fuchsia", target_arch = "aarch64")
)))]}
type_alias! { "schar.md", c_schar = i8, NonZero_c_schar = NonZeroI8; }
type_alias! { "uchar.md", c_uchar = u8, NonZero_c_uchar = NonZeroU8; }
type_alias! { "short.md", c_short = i16, NonZero_c_short = NonZeroI16; }
type_alias! { "ushort.md", c_ushort = u16, NonZero_c_ushort = NonZeroU16; }
type_alias! { "int.md", c_int = i32, NonZero_c_int = NonZeroI32; }
type_alias! { "uint.md", c_uint = u32, NonZero_c_uint = NonZeroU32; }
type_alias! { "long.md", c_long = i32, NonZero_c_long = NonZeroI32;
#[cfg(any(target_pointer_width = "32", windows))] }
type_alias! { "ulong.md", c_ulong = u32, NonZero_c_ulong = NonZeroU32;
#[cfg(any(target_pointer_width = "32", windows))] }
type_alias! { "long.md", c_long = i64, NonZero_c_long = NonZeroI64;
#[cfg(all(target_pointer_width = "64", not(windows)))] }
type_alias! { "ulong.md", c_ulong = u64, NonZero_c_ulong = NonZeroU64;
#[cfg(all(target_pointer_width = "64", not(windows)))] }
type_alias! { "longlong.md", c_longlong = i64, NonZero_c_longlong = NonZeroI64; }
type_alias! { "ulonglong.md", c_ulonglong = u64, NonZero_c_ulonglong = NonZeroU64; }
type_alias_no_nz! { "float.md", c_float = f32; }
type_alias_no_nz! { "double.md", c_double = f64; }

#[stable(feature = "raw_os", since = "1.1.0")]
#[doc(no_inline)]
pub use core::ffi::c_void;

/// Equivalent to C's `size_t` type, from `stddef.h` (or `cstddef` for C++).
///
/// This type is currently always [`usize`], however in the future there may be
/// platforms where this is not the case.
#[unstable(feature = "c_size_t", issue = "none")]
pub type c_size_t = usize;

/// Equivalent to C's `ssize_t` type, from `stddef.h` (or `cstddef` for C++).
///
/// This type is currently always [`isize`], however in the future there may be
/// platforms where this is not the case.
#[unstable(feature = "c_size_t", issue = "none")]
pub type c_ssize_t = isize;
