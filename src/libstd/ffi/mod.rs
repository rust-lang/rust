// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This module provides utilities to handle C-like strings.  It is
//! mainly of use for FFI (Foreign Function Interface) bindings and
//! code that needs to exchange C-like strings with other languages.
//!
//! # Overview
//!
//! Rust represents owned strings with the [`String`] type, and
//! borrowed slices of strings with the [`str`] primitive.  Both are
//! always in UTF-8 encoding, and may contain nul bytes in the middle,
//! i.e. if you look at the bytes that make up the string, there may
//! be a `0` among them.  Both `String` and `str` know their length;
//! there are no nul terminators at the end of strings like in C.
//!
//! C strings are different from Rust strings:
//!
//! * **Encodings** - C strings may have different encodings.  If
//! you are bringing in strings from C APIs, you should check what
//! encoding you are getting.  Rust strings are always UTF-8.
//!
//! * **Character width** - C strings may use "normal" or "wide"
//! characters, i.e. `char` or `wchar_t`, respectively.  The C
//! standard leaves the actual sizes of those types open to
//! interpretation, but defines different APIs for strings made up of
//! each character type.  Rust strings are always UTF-8, so different
//! Unicode characters will be encoded in a variable number of bytes
//! each.  The Rust type [`char`] represents a '[Unicode
//! scalar value]', which is similar to, but not the same as, a
//! '[Unicode code point]'.
//!
//! * **Nul terminators and implicit string lengths** - Often, C
//! strings are nul-terminated, i.e. they have a `0` character at the
//! end.  The length of a string buffer is not known *a priori*;
//! instead, to compute the length of a string, C code must manually
//! call a function like `strlen()` for `char`-based strings, or
//! `wcslen()` for `wchar_t`-based ones.  Those functions return the
//! number of characters in the string excluding the nul terminator,
//! so the buffer length is really `len+1` characters.  Rust strings
//! don't have a nul terminator, and they always know their length.
//!
//! * **No nul characters in the middle of the string** - When C
//! strings have a nul terminator character, this usually means that
//! they cannot have nul characters in the middle — a nul character
//! would essentially truncate the string.  Rust strings *can* have
//! nul characters in the middle, since they don't use nul
//! terminators.
//!
//! # Representations of non-Rust strings
//!
//! [`CString`] and [`CStr`] are useful when you need to transfer
//! UTF-8 strings to and from C, respectively:
//!
//! * **From Rust to C:** [`CString`] represents an owned, C-friendly
//! UTF-8 string:  it is valid UTF-8, it is nul-terminated, and has no
//! nul characters in the middle.  Rust code can create a `CString`
//! out of a normal string (provided that the string doesn't have nul
//! characters in the middle), and then use a variety of methods to
//! obtain a raw `*mut u8` that can then be passed as an argument to C
//! functions.
//!
//! * **From C to Rust:** [`CStr`] represents a borrowed C string; it
//! is what you would use to wrap a raw `*const u8` that you got from
//! a C function.  A `CStr` is just guaranteed to be a nul-terminated
//! array of bytes; the UTF-8 validation step only happens when you
//! request to convert it to a `&str`.
//!
//! [`OsString`] and [`OsStr`] are useful when you need to transfer
//! strings to and from operating system calls.  If you need Rust
//! strings out of them, they can take care of conversion to and from
//! the operating system's preferred form for strings — of course, it
//! may not be possible to convert all valid operating system strings
//! into valid UTF-8; the `OsString` and `OsStr` functions let you know
//! when this is the case.
//!
//! * [`OsString`] represents an owned string in whatever
//! representation the operating system prefers.  In the Rust standard
//! library, various APIs that transfer strings to/from the operating
//! system use `OsString` instead of plain strings.  For example,
//! [`env::var_os()`] is used to query environment variables; it
//! returns an `Option<OsString>`.  If the environment variable exists
//! you will get a `Some(os_string)`, which you can *then* try to
//! convert to a Rust string.  This yields a [`Result<>`], so that
//! your code can detect errors in case the environment variable did
//! not in fact contain valid Unicode data.
//!
//! * [`OsStr`] represents a borrowed reference to a string in a format that
//! can be passed to the operating system.  It can be converted into
//! an UTF-8 Rust string slice in a similar way to `OsString`.
//!
//! [`String`]: ../string/struct.String.html
//! [`str`]: ../primitive.str.html
//! [`char`]: ../primitive.char.html
//! [Unicode scalar value]: http://www.unicode.org/glossary/#unicode_scalar_value
//! [Unicode code point]: http://www.unicode.org/glossary/#code_point
//! [`CString`]: struct.CString.html
//! [`CStr`]: struct.CStr.html
//! [`OsString`]: struct.OsString.html
//! [`OsStr`]: struct.OsStr.html
//! [`env::set_var()`]: ../env/fn.set_var.html
//! [`env::var_os()`]: ../env/fn.var_os.html
//! [`Result<>`]: ../result/enum.Result.html

#![stable(feature = "rust1", since = "1.0.0")]

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::c_str::{CString, CStr, NulError, IntoStringError};
#[stable(feature = "cstr_from_bytes", since = "1.10.0")]
pub use self::c_str::{FromBytesWithNulError};

#[stable(feature = "rust1", since = "1.0.0")]
pub use self::os_str::{OsString, OsStr};

mod c_str;
mod os_str;
