//! Windows-specific extensions to primitives in the [`std::ffi`] module.
//!
//! # Overview
//!
//! For historical reasons, the Windows API uses a form of potentially
//! ill-formed UTF-16 encoding for strings. Specifically, the 16-bit
//! code units in Windows strings may contain [isolated surrogate code
//! points which are not paired together][ill-formed-utf-16]. The
//! Unicode standard requires that surrogate code points (those in the
//! range U+D800 to U+DFFF) always be *paired*, because in the UTF-16
//! encoding a *surrogate code unit pair* is used to encode a single
//! character. For compatibility with code that does not enforce
//! these pairings, Windows does not enforce them, either.
//!
//! While it is not always possible to convert such a string losslessly into
//! a valid UTF-16 string (or even UTF-8), it is often desirable to be
//! able to round-trip such a string from and to Windows APIs
//! losslessly. For example, some Rust code may be "bridging" some
//! Windows APIs together, just passing `WCHAR` strings among those
//! APIs without ever really looking into the strings.
//!
//! If Rust code *does* need to look into those strings, it can
//! convert them to valid UTF-8, possibly lossily, by substituting
//! invalid sequences with [`U+FFFD REPLACEMENT CHARACTER`][U+FFFD], as is
//! conventionally done in other Rust APIs that deal with string
//! encodings.
//!
//! # `OsStringExt` and `OsStrExt`
//!
//! [`OsString`] is the Rust wrapper for owned strings in the
//! preferred representation of the operating system. On Windows,
//! this struct gets augmented with an implementation of the
//! [`OsStringExt`] trait, which has an [`OsStringExt::from_wide`] method. This
//! lets you create an [`OsString`] from a `&[u16]` slice; presumably
//! you get such a slice out of a `WCHAR` Windows API.
//!
//! Similarly, [`OsStr`] is the Rust wrapper for borrowed strings from
//! preferred representation of the operating system. On Windows, the
//! [`OsStrExt`] trait provides the [`OsStrExt::encode_wide`] method, which
//! outputs an [`EncodeWide`] iterator. You can [`collect`] this
//! iterator, for example, to obtain a `Vec<u16>`; you can later get a
//! pointer to this vector's contents and feed it to Windows APIs.
//!
//! These traits, along with [`OsString`] and [`OsStr`], work in
//! conjunction so that it is possible to **round-trip** strings from
//! Windows and back, with no loss of data, even if the strings are
//! ill-formed UTF-16.
//!
//! [ill-formed-utf-16]: https://simonsapin.github.io/wtf-8/#ill-formed-utf-16
//! [`collect`]: crate::iter::Iterator::collect
//! [U+FFFD]: crate::char::REPLACEMENT_CHARACTER
//! [`std::ffi`]: crate::ffi

#![stable(feature = "rust1", since = "1.0.0")]

#[stable(feature = "rust1", since = "1.0.0")]
pub use alloc::ffi::os_str::os_str_ext_windows::OsStringExt;
#[stable(feature = "rust1", since = "1.0.0")]
pub use core::ffi::os_str::os_str_ext_windows::{EncodeWide, OsStrExt};
