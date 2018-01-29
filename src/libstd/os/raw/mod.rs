// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Platform-specific types, as defined by C.
//!
//! Code that interacts via FFI will almost certainly be using the
//! base types provided by C, which aren't nearly as nicely defined
//! as Rust's primitive types. This module provides types which will
//! match those defined by C, so that code that interacts with C will
//! refer to the correct types.

#![stable(feature = "raw_os", since = "1.1.0")]

use fmt;

#[doc(include = "os/raw/char.md")]
#[cfg(any(all(target_os = "linux", any(target_arch = "aarch64",
                                       target_arch = "arm",
                                       target_arch = "powerpc",
                                       target_arch = "powerpc64",
                                       target_arch = "s390x")),
          all(target_os = "android", any(target_arch = "aarch64",
                                         target_arch = "arm")),
          all(target_os = "l4re", target_arch = "x86_64"),
          all(target_os = "openbsd", target_arch = "aarch64"),
          all(target_os = "fuchsia", target_arch = "aarch64")))]
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_char = u8;
#[doc(include = "os/raw/char.md")]
#[cfg(not(any(all(target_os = "linux", any(target_arch = "aarch64",
                                           target_arch = "arm",
                                           target_arch = "powerpc",
                                           target_arch = "powerpc64",
                                           target_arch = "s390x")),
              all(target_os = "android", any(target_arch = "aarch64",
                                             target_arch = "arm")),
              all(target_os = "l4re", target_arch = "x86_64"),
              all(target_os = "openbsd", target_arch = "aarch64"),
              all(target_os = "fuchsia", target_arch = "aarch64"))))]
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_char = i8;
#[doc(include = "os/raw/schar.md")]
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_schar = i8;
#[doc(include = "os/raw/uchar.md")]
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_uchar = u8;
#[doc(include = "os/raw/short.md")]
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_short = i16;
#[doc(include = "os/raw/ushort.md")]
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_ushort = u16;
#[doc(include = "os/raw/int.md")]
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_int = i32;
#[doc(include = "os/raw/uint.md")]
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_uint = u32;
#[doc(include = "os/raw/long.md")]
#[cfg(any(target_pointer_width = "32", windows))]
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_long = i32;
#[doc(include = "os/raw/ulong.md")]
#[cfg(any(target_pointer_width = "32", windows))]
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_ulong = u32;
#[doc(include = "os/raw/long.md")]
#[cfg(all(target_pointer_width = "64", not(windows)))]
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_long = i64;
#[doc(include = "os/raw/ulong.md")]
#[cfg(all(target_pointer_width = "64", not(windows)))]
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_ulong = u64;
#[doc(include = "os/raw/longlong.md")]
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_longlong = i64;
#[doc(include = "os/raw/ulonglong.md")]
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_ulonglong = u64;
#[doc(include = "os/raw/float.md")]
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_float = f32;
#[doc(include = "os/raw/double.md")]
#[stable(feature = "raw_os", since = "1.1.0")] pub type c_double = f64;

/// Equivalent to C's `void` type when used as a [pointer].
///
/// In essence, `*const c_void` is equivalent to C's `const void*`
/// and `*mut c_void` is equivalent to C's `void*`. That said, this is
/// *not* the same as C's `void` return type, which is Rust's `()` type.
///
/// Ideally, this type would be equivalent to [`!`], but currently it may
/// be more ideal to use `c_void` for FFI purposes.
///
/// [`!`]: ../../primitive.never.html
/// [pointer]: ../../primitive.pointer.html
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

#[stable(feature = "std_debug", since = "1.16.0")]
impl fmt::Debug for c_void {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("c_void")
    }
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use any::TypeId;
    use libc;
    use mem;

    macro_rules! ok {
        ($($t:ident)*) => {$(
            assert!(TypeId::of::<libc::$t>() == TypeId::of::<raw::$t>(),
                    "{} is wrong", stringify!($t));
        )*}
    }

    #[test]
    fn same() {
        use os::raw;
        ok!(c_char c_schar c_uchar c_short c_ushort c_int c_uint c_long c_ulong
            c_longlong c_ulonglong c_float c_double);
    }
}
