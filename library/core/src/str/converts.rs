//! Ways to create a `str` from bytes slice.

use crate::mem;

use super::validations::run_utf8_validation;
use super::Utf8Error;

/// Converts a slice of bytes to a string slice.
///
/// A string slice ([`&str`]) is made of bytes ([`u8`]), and a byte slice
/// ([`&[u8]`][byteslice]) is made of bytes, so this function converts between
/// the two. Not all byte slices are valid string slices, however: [`&str`] requires
/// that it is valid UTF-8. `from_utf8()` checks to ensure that the bytes are valid
/// UTF-8, and then does the conversion.
///
/// [`&str`]: str
/// [byteslice]: slice
///
/// If you are sure that the byte slice is valid UTF-8, and you don't want to
/// incur the overhead of the validity check, there is an unsafe version of
/// this function, [`from_utf8_unchecked`], which has the same
/// behavior but skips the check.
///
/// If you need a `String` instead of a `&str`, consider
/// [`String::from_utf8`][string].
///
/// [string]: ../../std/string/struct.String.html#method.from_utf8
///
/// Because you can stack-allocate a `[u8; N]`, and you can take a
/// [`&[u8]`][byteslice] of it, this function is one way to have a
/// stack-allocated string. There is an example of this in the
/// examples section below.
///
/// [byteslice]: slice
///
/// # Errors
///
/// Returns `Err` if the slice is not UTF-8 with a description as to why the
/// provided slice is not UTF-8.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::str;
///
/// // some bytes, in a vector
/// let sparkle_heart = vec![240, 159, 146, 150];
///
/// // We know these bytes are valid, so just use `unwrap()`.
/// let sparkle_heart = str::from_utf8(&sparkle_heart).unwrap();
///
/// assert_eq!("💖", sparkle_heart);
/// ```
///
/// Incorrect bytes:
///
/// ```
/// use std::str;
///
/// // some invalid bytes, in a vector
/// let sparkle_heart = vec![0, 159, 146, 150];
///
/// assert!(str::from_utf8(&sparkle_heart).is_err());
/// ```
///
/// See the docs for [`Utf8Error`] for more details on the kinds of
/// errors that can be returned.
///
/// A "stack allocated string":
///
/// ```
/// use std::str;
///
/// // some bytes, in a stack-allocated array
/// let sparkle_heart = [240, 159, 146, 150];
///
/// // We know these bytes are valid, so just use `unwrap()`.
/// let sparkle_heart = str::from_utf8(&sparkle_heart).unwrap();
///
/// assert_eq!("💖", sparkle_heart);
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub fn from_utf8(v: &[u8]) -> Result<&str, Utf8Error> {
    run_utf8_validation(v)?;
    // SAFETY: Just ran validation.
    Ok(unsafe { from_utf8_unchecked(v) })
}

/// Converts a mutable slice of bytes to a mutable string slice.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::str;
///
/// // "Hello, Rust!" as a mutable vector
/// let mut hellorust = vec![72, 101, 108, 108, 111, 44, 32, 82, 117, 115, 116, 33];
///
/// // As we know these bytes are valid, we can use `unwrap()`
/// let outstr = str::from_utf8_mut(&mut hellorust).unwrap();
///
/// assert_eq!("Hello, Rust!", outstr);
/// ```
///
/// Incorrect bytes:
///
/// ```
/// use std::str;
///
/// // Some invalid bytes in a mutable vector
/// let mut invalid = vec![128, 223];
///
/// assert!(str::from_utf8_mut(&mut invalid).is_err());
/// ```
/// See the docs for [`Utf8Error`] for more details on the kinds of
/// errors that can be returned.
#[stable(feature = "str_mut_extras", since = "1.20.0")]
pub fn from_utf8_mut(v: &mut [u8]) -> Result<&mut str, Utf8Error> {
    run_utf8_validation(v)?;
    // SAFETY: Just ran validation.
    Ok(unsafe { from_utf8_unchecked_mut(v) })
}

/// Converts a slice of bytes to a string slice without checking
/// that the string contains valid UTF-8.
///
/// See the safe version, [`from_utf8`], for more information.
///
/// # Safety
///
/// This function is unsafe because it does not check that the bytes passed to
/// it are valid UTF-8. If this constraint is violated, undefined behavior
/// results, as the rest of Rust assumes that [`&str`]s are valid UTF-8.
///
/// [`&str`]: str
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::str;
///
/// // some bytes, in a vector
/// let sparkle_heart = vec![240, 159, 146, 150];
///
/// let sparkle_heart = unsafe {
///     str::from_utf8_unchecked(&sparkle_heart)
/// };
///
/// assert_eq!("💖", sparkle_heart);
/// ```
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_const_stable(feature = "const_str_from_utf8_unchecked", since = "1.55.0")]
pub const unsafe fn from_utf8_unchecked(v: &[u8]) -> &str {
    // SAFETY: the caller must guarantee that the bytes `v` are valid UTF-8.
    // Also relies on `&str` and `&[u8]` having the same layout.
    unsafe { mem::transmute(v) }
}

/// Converts a slice of bytes to a string slice without checking
/// that the string contains valid UTF-8; mutable version.
///
/// See the immutable version, [`from_utf8_unchecked()`] for more information.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use std::str;
///
/// let mut heart = vec![240, 159, 146, 150];
/// let heart = unsafe { str::from_utf8_unchecked_mut(&mut heart) };
///
/// assert_eq!("💖", heart);
/// ```
#[inline]
#[stable(feature = "str_mut_extras", since = "1.20.0")]
pub unsafe fn from_utf8_unchecked_mut(v: &mut [u8]) -> &mut str {
    // SAFETY: the caller must guarantee that the bytes `v`
    // are valid UTF-8, thus the cast to `*mut str` is safe.
    // Also, the pointer dereference is safe because that pointer
    // comes from a reference which is guaranteed to be valid for writes.
    unsafe { &mut *(v as *mut [u8] as *mut str) }
}
