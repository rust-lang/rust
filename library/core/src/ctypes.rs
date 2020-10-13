//! Type aliases to C types like c_int for use with bindgen
//!
//! # MSRV
//!
//! This crate is guaranteed to compile on stable Rust 1.30.0 and up. It *might* compile with older
//! versions but that may change in any new patch release.

#![allow(non_camel_case_types)]
#![deny(warnings)]

// AD = Architecture dependent
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use ad::*;
// OD = OS dependent
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use od::*;
// OD IOVEC = OS dependent iovec
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use od_iovec::*;
// OD wchar = OS dependent wchar_t
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use od_wchar::*;
// OD wint = OS dependent wint_t
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use od_wint::*;
// PWD = Pointer Width Dependent
#[stable(feature = "core_primitive", since = "1.43.0")]
pub use pwd::*;

//===================================================================
// c_char, c_int, c_uint
//===================================================================

#[cfg(any(target_arch = "aarch64",
          target_arch = "arm",
          target_arch = "asmjs",
          target_arch = "wasm32",
          target_arch = "wasm64",
          target_arch = "powerpc",
          target_arch = "powerpc64",
          target_arch = "s390x",
          target_arch = "riscv32",
          target_arch = "riscv64"))]
mod ad {
    use super::c_uchar;
    #[stable(feature = "rust1", since = "1.0.0")]
    pub type c_char = c_uchar;

    #[stable(feature = "rust1", since = "1.0.0")]
    pub type c_int = i32;
    #[stable(feature = "rust1", since = "1.0.0")]
    pub type c_uint = u32;
}

#[cfg(any(target_arch = "mips",
          target_arch = "mips64",
          target_arch = "sparc64",
          target_arch = "x86",
          target_arch = "x86_64",
          target_arch = "nvptx",
          target_arch = "nvptx64",
          target_arch = "xtensa"))]
mod ad {
    use super::c_schar;
    #[stable(feature = "rust1", since = "1.0.0")]
    pub type c_char = c_schar;

    #[stable(feature = "rust1", since = "1.0.0")]
    pub type c_int = i32;
    #[stable(feature = "rust1", since = "1.0.0")]
    pub type c_uint = u32;
}

#[cfg(target_arch = "msp430")]
mod ad {
    use super::c_uchar;
    #[stable(feature = "rust1", since = "1.0.0")]
    pub type c_char = c_uchar;

    #[stable(feature = "rust1", since = "1.0.0")]
    pub type c_int = i16;
    #[stable(feature = "rust1", since = "1.0.0")]
    pub type c_uint = u16;
}

//===================================================================
// c_long, c_ulong
//===================================================================

// NOTE c_{,u}long definitions come from libc v0.2.3
#[cfg(not(
      any(target_os = "windows",
          target_os = "redox",
          target_os = "solaris")))]
mod od {
#[cfg(any(target_pointer_width = "16",
          target_pointer_width = "32"))]
#[stable(feature = "rust1", since = "1.0.0")]
pub type c_long = i32;
#[cfg(any(target_pointer_width = "16",
          target_pointer_width = "32"))]
#[stable(feature = "rust1", since = "1.0.0")]
pub type c_ulong = u32;

#[cfg(target_pointer_width = "64")]
#[stable(feature = "rust1", since = "1.0.0")]
pub type c_long = i64;
#[cfg(target_pointer_width = "64")]
#[stable(feature = "rust1", since = "1.0.0")]
pub type c_ulong = u64;
}

#[cfg(any(target_os = "windows"))]
#[stable(feature = "rust1", since = "1.0.0")]
mod od {
#[stable(feature = "rust1", since = "1.0.0")]
pub type c_long = i32;
#[stable(feature = "rust1", since = "1.0.0")]
pub type c_ulong = u32;
}

#[cfg(any(target_os = "redox",
          target_os = "solaris"))]
#[stable(feature = "rust1", since = "1.0.0")]
mod od {
#[stable(feature = "rust1", since = "1.0.0")]
pub type c_long = i64;
#[stable(feature = "rust1", since = "1.0.0")]
pub type c_ulong = u64;
}

//===================================================================
// int8_t to c_void, all platform are the same
//===================================================================

#[stable(feature = "rust1", since = "1.0.0")]
pub type int8_t = i8;
#[stable(feature = "rust1", since = "1.0.0")]
pub type int16_t = i16;
#[stable(feature = "rust1", since = "1.0.0")]
pub type int32_t = i32;
#[stable(feature = "rust1", since = "1.0.0")]
pub type int64_t = i64;

#[stable(feature = "rust1", since = "1.0.0")]
pub type uint8_t = u8;
#[stable(feature = "rust1", since = "1.0.0")]
pub type uint16_t = u16;
#[stable(feature = "rust1", since = "1.0.0")]
pub type uint32_t = u32;
#[stable(feature = "rust1", since = "1.0.0")]
pub type uint64_t = u64;

#[stable(feature = "rust1", since = "1.0.0")]
pub type c_schar = i8;
#[stable(feature = "rust1", since = "1.0.0")]
pub type c_short = i16;
#[stable(feature = "rust1", since = "1.0.0")]
pub type c_longlong = i64;

#[stable(feature = "rust1", since = "1.0.0")]
pub type c_uchar = u8;
#[stable(feature = "rust1", since = "1.0.0")]
pub type c_ushort = u16;
#[stable(feature = "rust1", since = "1.0.0")]
pub type c_ulonglong = u64;

#[stable(feature = "rust1", since = "1.0.0")]
pub type c_float = f32;
#[stable(feature = "rust1", since = "1.0.0")]
pub type c_double = f64;

#[stable(feature = "rust1", since = "1.0.0")]
pub type intmax_t = i64;
#[stable(feature = "rust1", since = "1.0.0")]
pub type uintmax_t = u64;

#[stable(feature = "rust1", since = "1.0.0")]
pub type size_t = usize;
#[stable(feature = "rust1", since = "1.0.0")]
pub type ptrdiff_t = isize;
#[stable(feature = "rust1", since = "1.0.0")]
pub type intptr_t = isize;
#[stable(feature = "rust1", since = "1.0.0")]
pub type uintptr_t = usize;
#[stable(feature = "rust1", since = "1.0.0")]
pub type ssize_t = isize;

#[stable(feature = "core_primitive", since = "1.43.0")]
pub type va_list<'a,'b> = super::ffi::VaList<'a, 'b>;

#[stable(feature = "rust1", since = "1.0.0")]
pub type c_void = crate::ffi::c_void;

//===================================================================
// struct iovec, iov_len_t
//===================================================================

// iovec is special to move part of std::io component into core, other
// struct type doesn't have such limitation.
// Add iov_len_t for easily do cast, becase windows are different

#[cfg(not(any(
          target_os = "windows")))]
mod od_iovec {
use super::size_t;
#[stable(feature = "rust1", since = "1.0.0")]
use super::c_void;
#[stable(feature = "rust1", since = "1.0.0")]
pub type iov_len_t = size_t;
/// Refer to https://pubs.opengroup.org/onlinepubs/009695399/basedefs/sys/uio.h.html
/// Refer to https://github.com/bminor/glibc/blob/master/misc/bits/types/struct_iovec.h
#[repr(C)]
#[derive(Copy, Clone,Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct iovec {
    #[stable(feature = "rust1", since = "1.0.0")]
    pub iov_base: *mut c_void,
    #[stable(feature = "rust1", since = "1.0.0")]
    pub iov_len: iov_len_t,
}
}

#[cfg(any(target_os = "windows"))]
mod od_iovec {
use super::c_ulong;
use super::c_void;
/// Refer to https://docs.microsoft.com/en-us/windows/win32/api/ws2def/ns-ws2def-wsabuf
#[stable(feature = "rust1", since = "1.0.0")]
pub type iov_len_t = c_ulong;
#[repr(C)]
#[derive(Copy, Clone,Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct iovec {
    #[stable(feature = "rust1", since = "1.0.0")]
    pub iov_len: iov_len_t,
    #[stable(feature = "rust1", since = "1.0.0")]
    pub iov_base: *mut c_void,
}
}

//===================================================================
// wchar_t
//===================================================================
#[cfg(not(
      any(target_os = "windows",
          target_os = "vxworks")))]
mod od_wchar {
use super::int32_t;
#[stable(feature = "rust1", since = "1.0.0")]
pub type wchar_t = int32_t;
}

#[cfg(any(target_os = "windows",
          target_os = "vxworks"))]
mod od_wchar {
use super::uint16_t;
#[stable(feature = "rust1", since = "1.0.0")]
pub type wchar_t = uint16_t;
}

//===================================================================
// wint_t
//===================================================================
#[cfg(not(
      any(target_os = "windows",
          target_os = "vxworks")))]
mod od_wint {
use super::uint32_t;
#[stable(feature = "rust1", since = "1.0.0")]
pub type wint_t = uint32_t;
#[stable(feature = "rust1", since = "1.0.0")]
pub const WEOF: wint_t = 0xFFFF_FFFFu32;
}

#[cfg(any(target_os = "windows"))]
mod od_wint {
use super::uint16_t;
#[stable(feature = "rust1", since = "1.0.0")]
pub type wint_t = uint16_t;
#[stable(feature = "rust1", since = "1.0.0")]
pub const WEOF: wint_t = 0xFFFFu16;
}

#[cfg(any(target_os = "vxworks"))]
mod od_wint {
use super::int32_t;
#[stable(feature = "rust1", since = "1.0.0")]
pub type wint_t = int32_t;
#[stable(feature = "rust1", since = "1.0.0")]
pub const WEOF: wint_t = -1i32;
}

//===================================================================
// wctype_t TODO: This is os dependent
//===================================================================
#[stable(feature = "rust1", since = "1.0.0")]
pub type wctype_t = i64;

//===================================================================
// pwd not used yet
//===================================================================

#[cfg(target_pointer_width = "32")]
mod pwd {}

#[cfg(target_pointer_width = "64")]
mod pwd {}

