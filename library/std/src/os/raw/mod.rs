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

macro_rules! type_alias {
    { $Docfile:tt, $Alias:ident = $Real:ty; $( $Cfg:tt )* } => {
        #[doc(include = $Docfile)]
        $( $Cfg )*
        #[stable(feature = "raw_os", since = "1.1.0")]
        pub type $Alias = $Real;
    }
}

type_alias! { "char.md", c_char = u8;
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
type_alias! { "char.md", c_char = i8;
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
type_alias! { "schar.md", c_schar = i8; }
type_alias! { "uchar.md", c_uchar = u8; }
type_alias! { "short.md", c_short = i16; }
type_alias! { "ushort.md", c_ushort = u16; }
type_alias! { "int.md", c_int = i32; }
type_alias! { "uint.md", c_uint = u32; }
type_alias! { "long.md", c_long = i32; #[cfg(any(target_pointer_width = "32", windows))] }
type_alias! { "ulong.md", c_ulong = u32; #[cfg(any(target_pointer_width = "32", windows))] }
type_alias! { "long.md", c_long = i64; #[cfg(all(target_pointer_width = "64", not(windows)))] }
type_alias! { "ulong.md", c_ulong = u64; #[cfg(all(target_pointer_width = "64", not(windows)))] }
type_alias! { "longlong.md", c_longlong = i64; }
type_alias! { "ulonglong.md", c_ulonglong = u64; }
type_alias! { "float.md", c_float = f32; }
type_alias! { "double.md", c_double = f64; }

#[stable(feature = "raw_os", since = "1.1.0")]
#[doc(no_inline)]
pub use core::ffi::c_void;
