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
//! [`String`]: crate::string::String
//! [`CStr`]: core::ffi::CStr

#![stable(feature = "alloc_ffi", since = "1.64.0")]

#[doc(inline)]
#[stable(feature = "alloc_c_string", since = "1.64.0")]
pub use self::c_str::CString;
#[doc(no_inline)]
#[stable(feature = "alloc_c_string", since = "1.64.0")]
pub use self::c_str::{FromVecWithNulError, IntoStringError, NulError};

#[unstable(feature = "c_str_module", issue = "112134")]
pub mod c_str;
