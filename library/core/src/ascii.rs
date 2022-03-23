//! Operations on ASCII strings and characters.
//!
//! Most string operations in Rust act on UTF-8 strings. However, at times it
//! makes more sense to only consider the ASCII character set for a specific
//! operation.
//!
//! The [`escape_default`] function provides an iterator over the bytes of an
//! escaped version of the character given.
#![stable(feature = "core_ascii", since = "1.26.0")]
use core::num::EscapeAscii;

/// An iterator over the escaped version of a byte.
///
/// This `struct` is created by the [`u8::escape_ascii`] method. See its
/// documentation for more information.
#[stable(feature = "rust1", since = "1.0.0")]
pub type EscapeDefault = EscapeAscii;

/// This is a re-export of the [`u8::escape_ascii`] method as a standalone function.
///
/// Since this function was stabilized before the inherent version, it is left here for
/// backwards compatibility. However, new code should reference [`u8::escape_ascii`]
/// instead.
#[stable(feature = "rust1", since = "1.0.0")]
#[inline]
pub fn escape_default(c: u8) -> EscapeDefault {
    c.escape_ascii()
}
