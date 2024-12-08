//! Utilities related to FFI bindings.
//!
//! This module provides utilities to handle data across non-Rust
//! interfaces, like other programming languages and the underlying
//! operating system. It is mainly of use for FFI (Foreign Function
//! Interface) bindings and code that needs to exchange C-like strings
//! with other languages.
//!
//! # Overview
//!
//! Rust represents owned strings with the [`String`] type, and
//! borrowed slices of strings with the [`str`] primitive. Both are
//! always in UTF-8 encoding, and may contain nul bytes in the middle,
//! i.e., if you look at the bytes that make up the string, there may
//! be a `\0` among them. Both `String` and `str` store their length
//! explicitly; there are no nul terminators at the end of strings
//! like in C.
//!
//! C strings are different from Rust strings:
//!
//! * **Encodings** - Rust strings are UTF-8, but C strings may use
//! other encodings. If you are using a string from C, you should
//! check its encoding explicitly, rather than just assuming that it
//! is UTF-8 like you can do in Rust.
//!
//! * **Character size** - C strings may use `char` or `wchar_t`-sized
//! characters; please **note** that C's `char` is different from Rust's.
//! The C standard leaves the actual sizes of those types open to
//! interpretation, but defines different APIs for strings made up of
//! each character type. Rust strings are always UTF-8, so different
//! Unicode characters will be encoded in a variable number of bytes
//! each. The Rust type [`char`] represents a '[Unicode scalar
//! value]', which is similar to, but not the same as, a '[Unicode
//! code point]'.
//!
//! * **Nul terminators and implicit string lengths** - Often, C
//! strings are nul-terminated, i.e., they have a `\0` character at the
//! end. The length of a string buffer is not stored, but has to be
//! calculated; to compute the length of a string, C code must
//! manually call a function like `strlen()` for `char`-based strings,
//! or `wcslen()` for `wchar_t`-based ones. Those functions return
//! the number of characters in the string excluding the nul
//! terminator, so the buffer length is really `len+1` characters.
//! Rust strings don't have a nul terminator; their length is always
//! stored and does not need to be calculated. While in Rust
//! accessing a string's length is an *O*(1) operation (because the
//! length is stored); in C it is an *O*(*n*) operation because the
//! length needs to be computed by scanning the string for the nul
//! terminator.
//!
//! * **Internal nul characters** - When C strings have a nul
//! terminator character, this usually means that they cannot have nul
//! characters in the middle — a nul character would essentially
//! truncate the string. Rust strings *can* have nul characters in
//! the middle, because nul does not have to mark the end of the
//! string in Rust.
//!
//! # Representations of non-Rust strings
//!
//! [`CString`] and [`CStr`] are useful when you need to transfer
//! UTF-8 strings to and from languages with a C ABI, like Python.
//!
//! * **From Rust to C:** [`CString`] represents an owned, C-friendly
//! string: it is nul-terminated, and has no internal nul characters.
//! Rust code can create a [`CString`] out of a normal string (provided
//! that the string doesn't have nul characters in the middle), and
//! then use a variety of methods to obtain a raw <code>\*mut [u8]</code> that can
//! then be passed as an argument to functions which use the C
//! conventions for strings.
//!
//! * **From C to Rust:** [`CStr`] represents a borrowed C string; it
//! is what you would use to wrap a raw <code>\*const [u8]</code> that you got from
//! a C function. A [`CStr`] is guaranteed to be a nul-terminated array
//! of bytes. Once you have a [`CStr`], you can convert it to a Rust
//! <code>&[str]</code> if it's valid UTF-8, or lossily convert it by adding
//! replacement characters.
//!
//! [`OsString`] and [`OsStr`] are useful when you need to transfer
//! strings to and from the operating system itself, or when capturing
//! the output of external commands. Conversions between [`OsString`],
//! [`OsStr`] and Rust strings work similarly to those for [`CString`]
//! and [`CStr`].
//!
//! * [`OsString`] losslessly represents an owned platform string. However, this
//! representation is not necessarily in a form native to the platform.
//! In the Rust standard library, various APIs that transfer strings to/from the operating
//! system use [`OsString`] instead of plain strings. For example,
//! [`env::var_os()`] is used to query environment variables; it
//! returns an <code>[Option]<[OsString]></code>. If the environment variable
//! exists you will get a <code>[Some]\(os_string)</code>, which you can
//! *then* try to convert to a Rust string. This yields a [`Result`], so that
//! your code can detect errors in case the environment variable did
//! not in fact contain valid Unicode data.
//!
//! * [`OsStr`] losslessly represents a borrowed reference to a platform string.
//! However, this representation is not necessarily in a form native to the platform.
//! It can be converted into a UTF-8 Rust string slice in a similar way to
//! [`OsString`].
//!
//! # Conversions
//!
//! ## On Unix
//!
//! On Unix, [`OsStr`] implements the
//! <code>std::os::unix::ffi::[OsStrExt][unix.OsStrExt]</code> trait, which
//! augments it with two methods, [`from_bytes`] and [`as_bytes`].
//! These do inexpensive conversions from and to byte slices.
//!
//! Additionally, on Unix [`OsString`] implements the
//! <code>std::os::unix::ffi::[OsStringExt][unix.OsStringExt]</code> trait,
//! which provides [`from_vec`] and [`into_vec`] methods that consume
//! their arguments, and take or produce vectors of [`u8`].
//!
//! ## On Windows
//!
//! An [`OsStr`] can be losslessly converted to a native Windows string. And
//! a native Windows string can be losslessly converted to an [`OsString`].
//!
//! On Windows, [`OsStr`] implements the
//! <code>std::os::windows::ffi::[OsStrExt][windows.OsStrExt]</code> trait,
//! which provides an [`encode_wide`] method. This provides an
//! iterator that can be [`collect`]ed into a vector of [`u16`]. After a nul
//! characters is appended, this is the same as a native Windows string.
//!
//! Additionally, on Windows [`OsString`] implements the
//! <code>std::os::windows:ffi::[OsStringExt][windows.OsStringExt]</code>
//! trait, which provides a [`from_wide`] method to convert a native Windows
//! string (without the terminating nul character) to an [`OsString`].
//!
//! ## Other platforms
//!
//! Many other platforms provide their own extension traits in a
//! `std::os::*::ffi` module.
//!
//! ## On all platforms
//!
//! On all platforms, [`OsStr`] consists of a sequence of bytes that is encoded as a superset of
//! UTF-8; see [`OsString`] for more details on its encoding on different platforms.
//!
//! For limited, inexpensive conversions from and to bytes, see [`OsStr::as_encoded_bytes`] and
//! [`OsStr::from_encoded_bytes_unchecked`].
//!
//! For basic string processing, see [`OsStr::slice_encoded_bytes`].
//!
//! [Unicode scalar value]: https://www.unicode.org/glossary/#unicode_scalar_value
//! [Unicode code point]: https://www.unicode.org/glossary/#code_point
//! [`env::set_var()`]: crate::env::set_var "env::set_var"
//! [`env::var_os()`]: crate::env::var_os "env::var_os"
//! [unix.OsStringExt]: crate::os::unix::ffi::OsStringExt "os::unix::ffi::OsStringExt"
//! [`from_vec`]: crate::os::unix::ffi::OsStringExt::from_vec "os::unix::ffi::OsStringExt::from_vec"
//! [`into_vec`]: crate::os::unix::ffi::OsStringExt::into_vec "os::unix::ffi::OsStringExt::into_vec"
//! [unix.OsStrExt]: crate::os::unix::ffi::OsStrExt "os::unix::ffi::OsStrExt"
//! [`from_bytes`]: crate::os::unix::ffi::OsStrExt::from_bytes "os::unix::ffi::OsStrExt::from_bytes"
//! [`as_bytes`]: crate::os::unix::ffi::OsStrExt::as_bytes "os::unix::ffi::OsStrExt::as_bytes"
//! [`OsStrExt`]: crate::os::unix::ffi::OsStrExt "os::unix::ffi::OsStrExt"
//! [windows.OsStrExt]: crate::os::windows::ffi::OsStrExt "os::windows::ffi::OsStrExt"
//! [`encode_wide`]: crate::os::windows::ffi::OsStrExt::encode_wide "os::windows::ffi::OsStrExt::encode_wide"
//! [`collect`]: crate::iter::Iterator::collect "iter::Iterator::collect"
//! [windows.OsStringExt]: crate::os::windows::ffi::OsStringExt "os::windows::ffi::OsStringExt"
//! [`from_wide`]: crate::os::windows::ffi::OsStringExt::from_wide "os::windows::ffi::OsStringExt::from_wide"

#![stable(feature = "rust1", since = "1.0.0")]

#[unstable(feature = "c_str_module", issue = "112134")]
pub mod c_str;

#[stable(feature = "core_c_void", since = "1.30.0")]
pub use core::ffi::c_void;
#[unstable(
    feature = "c_variadic",
    reason = "the `c_variadic` feature has not been properly tested on \
              all supported platforms",
    issue = "44930"
)]
pub use core::ffi::{VaList, VaListImpl};
#[stable(feature = "core_ffi_c", since = "1.64.0")]
pub use core::ffi::{
    c_char, c_double, c_float, c_int, c_long, c_longlong, c_schar, c_short, c_uchar, c_uint,
    c_ulong, c_ulonglong, c_ushort,
};

#[doc(no_inline)]
#[stable(feature = "cstr_from_bytes_until_nul", since = "1.69.0")]
pub use self::c_str::FromBytesUntilNulError;
#[doc(no_inline)]
#[stable(feature = "cstr_from_bytes", since = "1.10.0")]
pub use self::c_str::FromBytesWithNulError;
#[doc(no_inline)]
#[stable(feature = "cstring_from_vec_with_nul", since = "1.58.0")]
pub use self::c_str::FromVecWithNulError;
#[doc(no_inline)]
#[stable(feature = "cstring_into", since = "1.7.0")]
pub use self::c_str::IntoStringError;
#[doc(no_inline)]
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::c_str::NulError;
#[doc(inline)]
#[stable(feature = "rust1", since = "1.0.0")]
pub use self::c_str::{CStr, CString};
#[stable(feature = "rust1", since = "1.0.0")]
#[doc(inline)]
pub use self::os_str::{OsStr, OsString};

#[unstable(feature = "os_str_display", issue = "120048")]
pub mod os_str;
