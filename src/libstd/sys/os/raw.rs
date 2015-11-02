// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Raw OS-specific types for the current platform/architecture

#![stable(feature = "raw_os", since = "1.1.0")]

#[cfg(target_arch = "aarch64")]
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_char = u8;
#[cfg(not(target_arch = "aarch64"))]
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_char = i8;
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_schar = i8;
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_uchar = u8;
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_short = i16;
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_ushort = u16;
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_int = i32;
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_uint = u32;
#[cfg(any(target_pointer_width = "32", windows))]
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_long = i32;
#[cfg(any(target_pointer_width = "32", windows))]
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_ulong = u32;
#[cfg(all(target_pointer_width = "64", not(windows)))]
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_long = i64;
#[cfg(all(target_pointer_width = "64", not(windows)))]
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_ulong = u64;
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_longlong = i64;
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_ulonglong = u64;
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_float = f32;
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_double = f64;

/// Type used to construct void pointers for use with C.
///
/// This type is only useful as a pointer target. Do not use it as a
/// return type for FFI functions which have the `void` return type in
/// C. Use the unit type `()` or omit the return type instead.
// NB: For LLVM to recognize the void pointer type and by extension
//     functions like malloc(), we need to have it represented as i8* in
//     LLVM bitcode. The enum used here ensures this and prevents misuse
//     of the "raw" type by only having private variants.. We need two
//     variants, because the compiler complains about the repr attribute
//     otherwise.
#[repr(u8)]
#[stable(feature = "raw_os", since = "1.1.0")]
pub enum c_void {
    #[unstable(feature = "c_void_variant", reason = "should not have to exist",
               issue = "0")]
    #[doc(hidden)] __variant1,
    #[unstable(feature = "c_void_variant", reason = "should not have to exist",
               issue = "0")]
    #[doc(hidden)] __variant2,
}

#[cfg(test)]
mod tests {
    use core::mem;
    use core::any::TypeId;

    macro_rules! ok {
        ($($t:ident)*) => {$(
            assert!(TypeId::of::<libc::$t>() == TypeId::of::<raw::$t>(),
                    "{} is wrong", stringify!($t));
        )*}
    }

    macro_rules! ok_size {
        ($($t:ident)*) => {$(
            assert!(mem::size_of::<libc::$t>() == mem::size_of::<raw::$t>(),
                    "{} is wrong", stringify!($t));
        )*}
    }

    #[test]
    #[cfg(any(unix, windows))]
    fn same() {
        use libc;

        use os::raw;
        ok!(c_char c_schar c_uchar c_short c_ushort c_int c_uint c_long c_ulong
            c_longlong c_ulonglong c_float c_double);
    }

    #[cfg(unix)]
    fn unix() {
        use libc;

        {
            use os::unix::raw;
            ok!(uid_t gid_t dev_t ino_t mode_t nlink_t off_t blksize_t blkcnt_t);
        }
        {
            use unix::platform::raw;
            ok_size!(stat);
        }
    }

    #[cfg(windows)]
    fn windows() {
        use os::windows::raw;
    }
}
