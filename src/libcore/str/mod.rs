// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! String manipulation
//!
//! For more details, see std::str

#![stable(feature = "rust1", since = "1.0.0")]

use self::pattern::Pattern;
use self::pattern::{Searcher, ReverseSearcher, DoubleEndedSearcher};

use char;
use convert::TryFrom;
use fmt;
use iter::{Map, Cloned, FusedIterator};
use mem;
use slice;

pub mod pattern;

/// A trait to abstract the idea of creating a new instance of a type from a
/// string.
///
/// `FromStr`'s [`from_str`] method is often used implicitly, through
/// [`str`]'s [`parse`] method. See [`parse`]'s documentation for examples.
///
/// [`from_str`]: #tymethod.from_str
/// [`str`]: ../../std/primitive.str.html
/// [`parse`]: ../../std/primitive.str.html#method.parse
///
/// # Examples
///
/// Basic implementation of `FromStr` on an example `Point` type:
///
/// ```
/// use std::str::FromStr;
/// use std::num::ParseIntError;
///
/// #[derive(Debug, PartialEq)]
/// struct Point {
///     x: i32,
///     y: i32
/// }
///
/// impl FromStr for Point {
///     type Err = ParseIntError;
///
///     fn from_str(s: &str) -> Result<Self, Self::Err> {
///         let coords: Vec<&str> = s.trim_matches(|p| p == '(' || p == ')' )
///                                  .split(",")
///                                  .collect();
///
///         let x_fromstr = coords[0].parse::<i32>()?;
///         let y_fromstr = coords[1].parse::<i32>()?;
///
///         Ok(Point { x: x_fromstr, y: y_fromstr })
///     }
/// }
///
/// let p = Point::from_str("(1,2)");
/// assert_eq!(p.unwrap(), Point{ x: 1, y: 2} )
/// ```
#[stable(feature = "rust1", since = "1.0.0")]
pub trait FromStr: Sized {
    /// The associated error which can be returned from parsing.
    #[stable(feature = "rust1", since = "1.0.0")]
    type Err;

    /// Parses a string `s` to return a value of this type.
    ///
    /// If parsing succeeds, return the value inside `Ok`, otherwise
    /// when the string is ill-formatted return an error specific to the
    /// inside `Err`. The error type is specific to implementation of the trait.
    ///
    /// # Examples
    ///
    /// Basic usage with [`i32`][ithirtytwo], a type that implements `FromStr`:
    ///
    /// [ithirtytwo]: ../../std/primitive.i32.html
    ///
    /// ```
    /// use std::str::FromStr;
    ///
    /// let s = "5";
    /// let x = i32::from_str(s).unwrap();
    ///
    /// assert_eq!(5, x);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    fn from_str(s: &str) -> Result<Self, Self::Err>;
}

#[stable(feature = "rust1", since = "1.0.0")]
impl FromStr for bool {
    type Err = ParseBoolError;

    /// Parse a `bool` from a string.
    ///
    /// Yields a `Result<bool, ParseBoolError>`, because `s` may or may not
    /// actually be parseable.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::str::FromStr;
    ///
    /// assert_eq!(FromStr::from_str("true"), Ok(true));
    /// assert_eq!(FromStr::from_str("false"), Ok(false));
    /// assert!(<bool as FromStr>::from_str("not even a boolean").is_err());
    /// ```
    ///
    /// Note, in many cases, the `.parse()` method on `str` is more proper.
    ///
    /// ```
    /// assert_eq!("true".parse(), Ok(true));
    /// assert_eq!("false".parse(), Ok(false));
    /// assert!("not even a boolean".parse::<bool>().is_err());
    /// ```
    #[inline]
    fn from_str(s: &str) -> Result<bool, ParseBoolError> {
        match s {
            "true"  => Ok(true),
            "false" => Ok(false),
            _       => Err(ParseBoolError { _priv: () }),
        }
    }
}

/// An error returned when parsing a `bool` using [`from_str`] fails
///
/// [`from_str`]: ../../std/primitive.bool.html#method.from_str
#[derive(Debug, Clone, PartialEq, Eq)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct ParseBoolError { _priv: () }

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for ParseBoolError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        "provided string was not `true` or `false`".fmt(f)
    }
}

/*
Section: Creating a string
*/

/// Errors which can occur when attempting to interpret a sequence of `u8`
/// as a string.
///
/// As such, the `from_utf8` family of functions and methods for both `String`s
/// and `&str`s make use of this error, for example.
#[derive(Copy, Eq, PartialEq, Clone, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Utf8Error {
    valid_up_to: usize,
    error_len: Option<u8>,
}

impl Utf8Error {
    /// Returns the index in the given string up to which valid UTF-8 was
    /// verified.
    ///
    /// It is the maximum index such that `from_utf8(&input[..index])`
    /// would return `Ok(_)`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// use std::str;
    ///
    /// // some invalid bytes, in a vector
    /// let sparkle_heart = vec![0, 159, 146, 150];
    ///
    /// // std::str::from_utf8 returns a Utf8Error
    /// let error = str::from_utf8(&sparkle_heart).unwrap_err();
    ///
    /// // the second byte is invalid here
    /// assert_eq!(1, error.valid_up_to());
    /// ```
    #[stable(feature = "utf8_error", since = "1.5.0")]
    pub fn valid_up_to(&self) -> usize { self.valid_up_to }

    /// Provide more information about the failure:
    ///
    /// * `None`: the end of the input was reached unexpectedly.
    ///   `self.valid_up_to()` is 1 to 3 bytes from the end of the input.
    ///   If a byte stream (such as a file or a network socket) is being decoded incrementally,
    ///   this could be a valid `char` whose UTF-8 byte sequence is spanning multiple chunks.
    ///
    /// * `Some(len)`: an unexpected byte was encountered.
    ///   The length provided is that of the invalid byte sequence
    ///   that starts at the index given by `valid_up_to()`.
    ///   Decoding should resume after that sequence
    ///   (after inserting a U+FFFD REPLACEMENT CHARACTER) in case of lossy decoding.
    #[unstable(feature = "utf8_error_error_len", reason ="new", issue = "40494")]
    pub fn error_len(&self) -> Option<usize> {
        self.error_len.map(|len| len as usize)
    }
}

/// Converts a slice of bytes to a string slice.
///
/// A string slice (`&str`) is made of bytes (`u8`), and a byte slice (`&[u8]`)
/// is made of bytes, so this function converts between the two. Not all byte
/// slices are valid string slices, however: `&str` requires that it is valid
/// UTF-8. `from_utf8()` checks to ensure that the bytes are valid UTF-8, and
/// then does the conversion.
///
/// If you are sure that the byte slice is valid UTF-8, and you don't want to
/// incur the overhead of the validity check, there is an unsafe version of
/// this function, [`from_utf8_unchecked`][fromutf8u], which has the same
/// behavior but skips the check.
///
/// [fromutf8u]: fn.from_utf8_unchecked.html
///
/// If you need a `String` instead of a `&str`, consider
/// [`String::from_utf8`][string].
///
/// [string]: ../../std/string/struct.String.html#method.from_utf8
///
/// Because you can stack-allocate a `[u8; N]`, and you can take a `&[u8]` of
/// it, this function is one way to have a stack-allocated string. There is
/// an example of this in the examples section below.
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
/// See the docs for [`Utf8Error`][error] for more details on the kinds of
/// errors that can be returned.
///
/// [error]: struct.Utf8Error.html
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
    Ok(unsafe { from_utf8_unchecked(v) })
}

/// Forms a str from a pointer and a length.
///
/// The `len` argument is the number of bytes in the string.
///
/// # Safety
///
/// This function is unsafe as there is no guarantee that the given pointer is
/// valid for `len` bytes, nor whether the lifetime inferred is a suitable
/// lifetime for the returned str.
///
/// The data must be valid UTF-8
///
/// `p` must be non-null, even for zero-length str.
///
/// # Caveat
///
/// The lifetime for the returned str is inferred from its usage. To
/// prevent accidental misuse, it's suggested to tie the lifetime to whichever
/// source lifetime is safe in the context, such as by providing a helper
/// function taking the lifetime of a host value for the str, or by explicit
/// annotation.
/// Performs the same functionality as `from_raw_parts`, except that a mutable
/// str is returned.
///
unsafe fn from_raw_parts_mut<'a>(p: *mut u8, len: usize) -> &'a mut str {
    mem::transmute::<&mut [u8], &mut str>(slice::from_raw_parts_mut(p, len))
}

/// Converts a slice of bytes to a string slice without checking
/// that the string contains valid UTF-8.
///
/// See the safe version, [`from_utf8`][fromutf8], for more information.
///
/// [fromutf8]: fn.from_utf8.html
///
/// # Safety
///
/// This function is unsafe because it does not check that the bytes passed to
/// it are valid UTF-8. If this constraint is violated, undefined behavior
/// results, as the rest of Rust assumes that [`&str`]s are valid UTF-8.
///
/// [`&str`]: ../../std/primitive.str.html
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
#[inline(always)]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn from_utf8_unchecked(v: &[u8]) -> &str {
    mem::transmute(v)
}

#[stable(feature = "rust1", since = "1.0.0")]
impl fmt::Display for Utf8Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(error_len) = self.error_len {
            write!(f, "invalid utf-8 sequence of {} bytes from index {}",
                   error_len, self.valid_up_to)
        } else {
            write!(f, "incomplete utf-8 byte sequence from index {}", self.valid_up_to)
        }
    }
}

/*
Section: Iterators
*/

/// Iterator for the char (representing *Unicode Scalar Values*) of a string.
///
/// Created with the method [`chars`].
///
/// [`chars`]: ../../std/primitive.str.html#method.chars
#[derive(Clone, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct Chars<'a> {
    iter: slice::Iter<'a, u8>
}

/// Returns the initial codepoint accumulator for the first byte.
/// The first byte is special, only want bottom 5 bits for width 2, 4 bits
/// for width 3, and 3 bits for width 4.
#[inline]
fn utf8_first_byte(byte: u8, width: u32) -> u32 { (byte & (0x7F >> width)) as u32 }

/// Returns the value of `ch` updated with continuation byte `byte`.
#[inline]
fn utf8_acc_cont_byte(ch: u32, byte: u8) -> u32 { (ch << 6) | (byte & CONT_MASK) as u32 }

/// Checks whether the byte is a UTF-8 continuation byte (i.e. starts with the
/// bits `10`).
#[inline]
fn utf8_is_cont_byte(byte: u8) -> bool { (byte & !CONT_MASK) == TAG_CONT_U8 }

#[inline]
fn unwrap_or_0(opt: Option<&u8>) -> u8 {
    match opt {
        Some(&byte) => byte,
        None => 0,
    }
}

/// Reads the next code point out of a byte iterator (assuming a
/// UTF-8-like encoding).
#[unstable(feature = "str_internals", issue = "0")]
#[inline]
pub fn next_code_point<'a, I: Iterator<Item = &'a u8>>(bytes: &mut I) -> Option<u32> {
    // Decode UTF-8
    let x = match bytes.next() {
        None => return None,
        Some(&next_byte) if next_byte < 128 => return Some(next_byte as u32),
        Some(&next_byte) => next_byte,
    };

    // Multibyte case follows
    // Decode from a byte combination out of: [[[x y] z] w]
    // NOTE: Performance is sensitive to the exact formulation here
    let init = utf8_first_byte(x, 2);
    let y = unwrap_or_0(bytes.next());
    let mut ch = utf8_acc_cont_byte(init, y);
    if x >= 0xE0 {
        // [[x y z] w] case
        // 5th bit in 0xE0 .. 0xEF is always clear, so `init` is still valid
        let z = unwrap_or_0(bytes.next());
        let y_z = utf8_acc_cont_byte((y & CONT_MASK) as u32, z);
        ch = init << 12 | y_z;
        if x >= 0xF0 {
            // [x y z w] case
            // use only the lower 3 bits of `init`
            let w = unwrap_or_0(bytes.next());
            ch = (init & 7) << 18 | utf8_acc_cont_byte(y_z, w);
        }
    }

    Some(ch)
}

/// Reads the last code point out of a byte iterator (assuming a
/// UTF-8-like encoding).
#[inline]
fn next_code_point_reverse<'a, I>(bytes: &mut I) -> Option<u32>
    where I: DoubleEndedIterator<Item = &'a u8>,
{
    // Decode UTF-8
    let w = match bytes.next_back() {
        None => return None,
        Some(&next_byte) if next_byte < 128 => return Some(next_byte as u32),
        Some(&back_byte) => back_byte,
    };

    // Multibyte case follows
    // Decode from a byte combination out of: [x [y [z w]]]
    let mut ch;
    let z = unwrap_or_0(bytes.next_back());
    ch = utf8_first_byte(z, 2);
    if utf8_is_cont_byte(z) {
        let y = unwrap_or_0(bytes.next_back());
        ch = utf8_first_byte(y, 3);
        if utf8_is_cont_byte(y) {
            let x = unwrap_or_0(bytes.next_back());
            ch = utf8_first_byte(x, 4);
            ch = utf8_acc_cont_byte(ch, y);
        }
        ch = utf8_acc_cont_byte(ch, z);
    }
    ch = utf8_acc_cont_byte(ch, w);

    Some(ch)
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for Chars<'a> {
    type Item = char;

    #[inline]
    fn next(&mut self) -> Option<char> {
        next_code_point(&mut self.iter).map(|ch| {
            // str invariant says `ch` is a valid Unicode Scalar Value
            unsafe {
                char::from_u32_unchecked(ch)
            }
        })
    }

    #[inline]
    fn count(self) -> usize {
        // length in `char` is equal to the number of non-continuation bytes
        let bytes_len = self.iter.len();
        let mut cont_bytes = 0;
        for &byte in self.iter {
            cont_bytes += utf8_is_cont_byte(byte) as usize;
        }
        bytes_len - cont_bytes
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.iter.len();
        // `(len + 3)` can't overflow, because we know that the `slice::Iter`
        // belongs to a slice in memory which has a maximum length of
        // `isize::MAX` (that's well below `usize::MAX`).
        ((len + 3) / 4, Some(len))
    }

    #[inline]
    fn last(mut self) -> Option<char> {
        // No need to go through the entire string.
        self.next_back()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> DoubleEndedIterator for Chars<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<char> {
        next_code_point_reverse(&mut self.iter).map(|ch| {
            // str invariant says `ch` is a valid Unicode Scalar Value
            unsafe {
                char::from_u32_unchecked(ch)
            }
        })
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl<'a> FusedIterator for Chars<'a> {}

impl<'a> Chars<'a> {
    /// View the underlying data as a subslice of the original data.
    ///
    /// This has the same lifetime as the original slice, and so the
    /// iterator can continue to be used while this exists.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut chars = "abc".chars();
    ///
    /// assert_eq!(chars.as_str(), "abc");
    /// chars.next();
    /// assert_eq!(chars.as_str(), "bc");
    /// chars.next();
    /// chars.next();
    /// assert_eq!(chars.as_str(), "");
    /// ```
    #[stable(feature = "iter_to_slice", since = "1.4.0")]
    #[inline]
    pub fn as_str(&self) -> &'a str {
        unsafe { from_utf8_unchecked(self.iter.as_slice()) }
    }
}

/// Iterator for a string's characters and their byte offsets.
#[derive(Clone, Debug)]
#[stable(feature = "rust1", since = "1.0.0")]
pub struct CharIndices<'a> {
    front_offset: usize,
    iter: Chars<'a>,
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for CharIndices<'a> {
    type Item = (usize, char);

    #[inline]
    fn next(&mut self) -> Option<(usize, char)> {
        let pre_len = self.iter.iter.len();
        match self.iter.next() {
            None => None,
            Some(ch) => {
                let index = self.front_offset;
                let len = self.iter.iter.len();
                self.front_offset += pre_len - len;
                Some((index, ch))
            }
        }
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.count()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn last(mut self) -> Option<(usize, char)> {
        // No need to go through the entire string.
        self.next_back()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> DoubleEndedIterator for CharIndices<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<(usize, char)> {
        match self.iter.next_back() {
            None => None,
            Some(ch) => {
                let index = self.front_offset + self.iter.iter.len();
                Some((index, ch))
            }
        }
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl<'a> FusedIterator for CharIndices<'a> {}

impl<'a> CharIndices<'a> {
    /// View the underlying data as a subslice of the original data.
    ///
    /// This has the same lifetime as the original slice, and so the
    /// iterator can continue to be used while this exists.
    #[stable(feature = "iter_to_slice", since = "1.4.0")]
    #[inline]
    pub fn as_str(&self) -> &'a str {
        self.iter.as_str()
    }
}

/// External iterator for a string's bytes.
/// Use with the `std::iter` module.
///
/// Created with the method [`bytes`].
///
/// [`bytes`]: ../../std/primitive.str.html#method.bytes
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone, Debug)]
pub struct Bytes<'a>(Cloned<slice::Iter<'a, u8>>);

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for Bytes<'a> {
    type Item = u8;

    #[inline]
    fn next(&mut self) -> Option<u8> {
        self.0.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.0.count()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.0.last()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        self.0.nth(n)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> DoubleEndedIterator for Bytes<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<u8> {
        self.0.next_back()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> ExactSizeIterator for Bytes<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl<'a> FusedIterator for Bytes<'a> {}

/// This macro generates a Clone impl for string pattern API
/// wrapper types of the form X<'a, P>
macro_rules! derive_pattern_clone {
    (clone $t:ident with |$s:ident| $e:expr) => {
        impl<'a, P: Pattern<'a>> Clone for $t<'a, P>
            where P::Searcher: Clone
        {
            fn clone(&self) -> Self {
                let $s = self;
                $e
            }
        }
    }
}

/// This macro generates two public iterator structs
/// wrapping a private internal one that makes use of the `Pattern` API.
///
/// For all patterns `P: Pattern<'a>` the following items will be
/// generated (generics omitted):
///
/// struct $forward_iterator($internal_iterator);
/// struct $reverse_iterator($internal_iterator);
///
/// impl Iterator for $forward_iterator
/// { /* internal ends up calling Searcher::next_match() */ }
///
/// impl DoubleEndedIterator for $forward_iterator
///       where P::Searcher: DoubleEndedSearcher
/// { /* internal ends up calling Searcher::next_match_back() */ }
///
/// impl Iterator for $reverse_iterator
///       where P::Searcher: ReverseSearcher
/// { /* internal ends up calling Searcher::next_match_back() */ }
///
/// impl DoubleEndedIterator for $reverse_iterator
///       where P::Searcher: DoubleEndedSearcher
/// { /* internal ends up calling Searcher::next_match() */ }
///
/// The internal one is defined outside the macro, and has almost the same
/// semantic as a DoubleEndedIterator by delegating to `pattern::Searcher` and
/// `pattern::ReverseSearcher` for both forward and reverse iteration.
///
/// "Almost", because a `Searcher` and a `ReverseSearcher` for a given
/// `Pattern` might not return the same elements, so actually implementing
/// `DoubleEndedIterator` for it would be incorrect.
/// (See the docs in `str::pattern` for more details)
///
/// However, the internal struct still represents a single ended iterator from
/// either end, and depending on pattern is also a valid double ended iterator,
/// so the two wrapper structs implement `Iterator`
/// and `DoubleEndedIterator` depending on the concrete pattern type, leading
/// to the complex impls seen above.
macro_rules! generate_pattern_iterators {
    {
        // Forward iterator
        forward:
            $(#[$forward_iterator_attribute:meta])*
            struct $forward_iterator:ident;

        // Reverse iterator
        reverse:
            $(#[$reverse_iterator_attribute:meta])*
            struct $reverse_iterator:ident;

        // Stability of all generated items
        stability:
            $(#[$common_stability_attribute:meta])*

        // Internal almost-iterator that is being delegated to
        internal:
            $internal_iterator:ident yielding ($iterty:ty);

        // Kind of delgation - either single ended or double ended
        delegate $($t:tt)*
    } => {
        $(#[$forward_iterator_attribute])*
        $(#[$common_stability_attribute])*
        pub struct $forward_iterator<'a, P: Pattern<'a>>($internal_iterator<'a, P>);

        $(#[$common_stability_attribute])*
        impl<'a, P: Pattern<'a>> fmt::Debug for $forward_iterator<'a, P>
            where P::Searcher: fmt::Debug
        {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.debug_tuple(stringify!($forward_iterator))
                    .field(&self.0)
                    .finish()
            }
        }

        $(#[$common_stability_attribute])*
        impl<'a, P: Pattern<'a>> Iterator for $forward_iterator<'a, P> {
            type Item = $iterty;

            #[inline]
            fn next(&mut self) -> Option<$iterty> {
                self.0.next()
            }
        }

        $(#[$common_stability_attribute])*
        impl<'a, P: Pattern<'a>> Clone for $forward_iterator<'a, P>
            where P::Searcher: Clone
        {
            fn clone(&self) -> Self {
                $forward_iterator(self.0.clone())
            }
        }

        $(#[$reverse_iterator_attribute])*
        $(#[$common_stability_attribute])*
        pub struct $reverse_iterator<'a, P: Pattern<'a>>($internal_iterator<'a, P>);

        $(#[$common_stability_attribute])*
        impl<'a, P: Pattern<'a>> fmt::Debug for $reverse_iterator<'a, P>
            where P::Searcher: fmt::Debug
        {
            fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.debug_tuple(stringify!($reverse_iterator))
                    .field(&self.0)
                    .finish()
            }
        }

        $(#[$common_stability_attribute])*
        impl<'a, P: Pattern<'a>> Iterator for $reverse_iterator<'a, P>
            where P::Searcher: ReverseSearcher<'a>
        {
            type Item = $iterty;

            #[inline]
            fn next(&mut self) -> Option<$iterty> {
                self.0.next_back()
            }
        }

        $(#[$common_stability_attribute])*
        impl<'a, P: Pattern<'a>> Clone for $reverse_iterator<'a, P>
            where P::Searcher: Clone
        {
            fn clone(&self) -> Self {
                $reverse_iterator(self.0.clone())
            }
        }

        #[unstable(feature = "fused", issue = "35602")]
        impl<'a, P: Pattern<'a>> FusedIterator for $forward_iterator<'a, P> {}

        #[unstable(feature = "fused", issue = "35602")]
        impl<'a, P: Pattern<'a>> FusedIterator for $reverse_iterator<'a, P>
            where P::Searcher: ReverseSearcher<'a> {}

        generate_pattern_iterators!($($t)* with $(#[$common_stability_attribute])*,
                                                $forward_iterator,
                                                $reverse_iterator, $iterty);
    };
    {
        double ended; with $(#[$common_stability_attribute:meta])*,
                           $forward_iterator:ident,
                           $reverse_iterator:ident, $iterty:ty
    } => {
        $(#[$common_stability_attribute])*
        impl<'a, P: Pattern<'a>> DoubleEndedIterator for $forward_iterator<'a, P>
            where P::Searcher: DoubleEndedSearcher<'a>
        {
            #[inline]
            fn next_back(&mut self) -> Option<$iterty> {
                self.0.next_back()
            }
        }

        $(#[$common_stability_attribute])*
        impl<'a, P: Pattern<'a>> DoubleEndedIterator for $reverse_iterator<'a, P>
            where P::Searcher: DoubleEndedSearcher<'a>
        {
            #[inline]
            fn next_back(&mut self) -> Option<$iterty> {
                self.0.next()
            }
        }
    };
    {
        single ended; with $(#[$common_stability_attribute:meta])*,
                           $forward_iterator:ident,
                           $reverse_iterator:ident, $iterty:ty
    } => {}
}

derive_pattern_clone!{
    clone SplitInternal
    with |s| SplitInternal { matcher: s.matcher.clone(), ..*s }
}

struct SplitInternal<'a, P: Pattern<'a>> {
    start: usize,
    end: usize,
    matcher: P::Searcher,
    allow_trailing_empty: bool,
    finished: bool,
}

impl<'a, P: Pattern<'a>> fmt::Debug for SplitInternal<'a, P> where P::Searcher: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SplitInternal")
            .field("start", &self.start)
            .field("end", &self.end)
            .field("matcher", &self.matcher)
            .field("allow_trailing_empty", &self.allow_trailing_empty)
            .field("finished", &self.finished)
            .finish()
    }
}

impl<'a, P: Pattern<'a>> SplitInternal<'a, P> {
    #[inline]
    fn get_end(&mut self) -> Option<&'a str> {
        if !self.finished && (self.allow_trailing_empty || self.end - self.start > 0) {
            self.finished = true;
            unsafe {
                let string = self.matcher.haystack().slice_unchecked(self.start, self.end);
                Some(string)
            }
        } else {
            None
        }
    }

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        if self.finished { return None }

        let haystack = self.matcher.haystack();
        match self.matcher.next_match() {
            Some((a, b)) => unsafe {
                let elt = haystack.slice_unchecked(self.start, a);
                self.start = b;
                Some(elt)
            },
            None => self.get_end(),
        }
    }

    #[inline]
    fn next_back(&mut self) -> Option<&'a str>
        where P::Searcher: ReverseSearcher<'a>
    {
        if self.finished { return None }

        if !self.allow_trailing_empty {
            self.allow_trailing_empty = true;
            match self.next_back() {
                Some(elt) if !elt.is_empty() => return Some(elt),
                _ => if self.finished { return None }
            }
        }

        let haystack = self.matcher.haystack();
        match self.matcher.next_match_back() {
            Some((a, b)) => unsafe {
                let elt = haystack.slice_unchecked(b, self.end);
                self.end = a;
                Some(elt)
            },
            None => unsafe {
                self.finished = true;
                Some(haystack.slice_unchecked(self.start, self.end))
            },
        }
    }
}

generate_pattern_iterators! {
    forward:
        /// Created with the method [`split`].
        ///
        /// [`split`]: ../../std/primitive.str.html#method.split
        struct Split;
    reverse:
        /// Created with the method [`rsplit`].
        ///
        /// [`rsplit`]: ../../std/primitive.str.html#method.rsplit
        struct RSplit;
    stability:
        #[stable(feature = "rust1", since = "1.0.0")]
    internal:
        SplitInternal yielding (&'a str);
    delegate double ended;
}

generate_pattern_iterators! {
    forward:
        /// Created with the method [`split_terminator`].
        ///
        /// [`split_terminator`]: ../../std/primitive.str.html#method.split_terminator
        struct SplitTerminator;
    reverse:
        /// Created with the method [`rsplit_terminator`].
        ///
        /// [`rsplit_terminator`]: ../../std/primitive.str.html#method.rsplit_terminator
        struct RSplitTerminator;
    stability:
        #[stable(feature = "rust1", since = "1.0.0")]
    internal:
        SplitInternal yielding (&'a str);
    delegate double ended;
}

derive_pattern_clone!{
    clone SplitNInternal
    with |s| SplitNInternal { iter: s.iter.clone(), ..*s }
}

struct SplitNInternal<'a, P: Pattern<'a>> {
    iter: SplitInternal<'a, P>,
    /// The number of splits remaining
    count: usize,
}

impl<'a, P: Pattern<'a>> fmt::Debug for SplitNInternal<'a, P> where P::Searcher: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("SplitNInternal")
            .field("iter", &self.iter)
            .field("count", &self.count)
            .finish()
    }
}

impl<'a, P: Pattern<'a>> SplitNInternal<'a, P> {
    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        match self.count {
            0 => None,
            1 => { self.count = 0; self.iter.get_end() }
            _ => { self.count -= 1; self.iter.next() }
        }
    }

    #[inline]
    fn next_back(&mut self) -> Option<&'a str>
        where P::Searcher: ReverseSearcher<'a>
    {
        match self.count {
            0 => None,
            1 => { self.count = 0; self.iter.get_end() }
            _ => { self.count -= 1; self.iter.next_back() }
        }
    }
}

generate_pattern_iterators! {
    forward:
        /// Created with the method [`splitn`].
        ///
        /// [`splitn`]: ../../std/primitive.str.html#method.splitn
        struct SplitN;
    reverse:
        /// Created with the method [`rsplitn`].
        ///
        /// [`rsplitn`]: ../../std/primitive.str.html#method.rsplitn
        struct RSplitN;
    stability:
        #[stable(feature = "rust1", since = "1.0.0")]
    internal:
        SplitNInternal yielding (&'a str);
    delegate single ended;
}

derive_pattern_clone!{
    clone MatchIndicesInternal
    with |s| MatchIndicesInternal(s.0.clone())
}

struct MatchIndicesInternal<'a, P: Pattern<'a>>(P::Searcher);

impl<'a, P: Pattern<'a>> fmt::Debug for MatchIndicesInternal<'a, P> where P::Searcher: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("MatchIndicesInternal")
            .field(&self.0)
            .finish()
    }
}

impl<'a, P: Pattern<'a>> MatchIndicesInternal<'a, P> {
    #[inline]
    fn next(&mut self) -> Option<(usize, &'a str)> {
        self.0.next_match().map(|(start, end)| unsafe {
            (start, self.0.haystack().slice_unchecked(start, end))
        })
    }

    #[inline]
    fn next_back(&mut self) -> Option<(usize, &'a str)>
        where P::Searcher: ReverseSearcher<'a>
    {
        self.0.next_match_back().map(|(start, end)| unsafe {
            (start, self.0.haystack().slice_unchecked(start, end))
        })
    }
}

generate_pattern_iterators! {
    forward:
        /// Created with the method [`match_indices`].
        ///
        /// [`match_indices`]: ../../std/primitive.str.html#method.match_indices
        struct MatchIndices;
    reverse:
        /// Created with the method [`rmatch_indices`].
        ///
        /// [`rmatch_indices`]: ../../std/primitive.str.html#method.rmatch_indices
        struct RMatchIndices;
    stability:
        #[stable(feature = "str_match_indices", since = "1.5.0")]
    internal:
        MatchIndicesInternal yielding ((usize, &'a str));
    delegate double ended;
}

derive_pattern_clone!{
    clone MatchesInternal
    with |s| MatchesInternal(s.0.clone())
}

struct MatchesInternal<'a, P: Pattern<'a>>(P::Searcher);

impl<'a, P: Pattern<'a>> fmt::Debug for MatchesInternal<'a, P> where P::Searcher: fmt::Debug {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("MatchesInternal")
            .field(&self.0)
            .finish()
    }
}

impl<'a, P: Pattern<'a>> MatchesInternal<'a, P> {
    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        self.0.next_match().map(|(a, b)| unsafe {
            // Indices are known to be on utf8 boundaries
            self.0.haystack().slice_unchecked(a, b)
        })
    }

    #[inline]
    fn next_back(&mut self) -> Option<&'a str>
        where P::Searcher: ReverseSearcher<'a>
    {
        self.0.next_match_back().map(|(a, b)| unsafe {
            // Indices are known to be on utf8 boundaries
            self.0.haystack().slice_unchecked(a, b)
        })
    }
}

generate_pattern_iterators! {
    forward:
        /// Created with the method [`matches`].
        ///
        /// [`matches`]: ../../std/primitive.str.html#method.matches
        struct Matches;
    reverse:
        /// Created with the method [`rmatches`].
        ///
        /// [`rmatches`]: ../../std/primitive.str.html#method.rmatches
        struct RMatches;
    stability:
        #[stable(feature = "str_matches", since = "1.2.0")]
    internal:
        MatchesInternal yielding (&'a str);
    delegate double ended;
}

/// Created with the method [`lines`].
///
/// [`lines`]: ../../std/primitive.str.html#method.lines
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone, Debug)]
pub struct Lines<'a>(Map<SplitTerminator<'a, char>, LinesAnyMap>);

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Iterator for Lines<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        self.0.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> DoubleEndedIterator for Lines<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a str> {
        self.0.next_back()
    }
}

#[unstable(feature = "fused", issue = "35602")]
impl<'a> FusedIterator for Lines<'a> {}

/// Created with the method [`lines_any`].
///
/// [`lines_any`]: ../../std/primitive.str.html#method.lines_any
#[stable(feature = "rust1", since = "1.0.0")]
#[rustc_deprecated(since = "1.4.0", reason = "use lines()/Lines instead now")]
#[derive(Clone, Debug)]
#[allow(deprecated)]
pub struct LinesAny<'a>(Lines<'a>);

/// A nameable, cloneable fn type
#[derive(Clone)]
struct LinesAnyMap;

impl<'a> Fn<(&'a str,)> for LinesAnyMap {
    #[inline]
    extern "rust-call" fn call(&self, (line,): (&'a str,)) -> &'a str {
        let l = line.len();
        if l > 0 && line.as_bytes()[l - 1] == b'\r' { &line[0 .. l - 1] }
        else { line }
    }
}

impl<'a> FnMut<(&'a str,)> for LinesAnyMap {
    #[inline]
    extern "rust-call" fn call_mut(&mut self, (line,): (&'a str,)) -> &'a str {
        Fn::call(&*self, (line,))
    }
}

impl<'a> FnOnce<(&'a str,)> for LinesAnyMap {
    type Output = &'a str;

    #[inline]
    extern "rust-call" fn call_once(self, (line,): (&'a str,)) -> &'a str {
        Fn::call(&self, (line,))
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated)]
impl<'a> Iterator for LinesAny<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        self.0.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
#[allow(deprecated)]
impl<'a> DoubleEndedIterator for LinesAny<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a str> {
        self.0.next_back()
    }
}

#[unstable(feature = "fused", issue = "35602")]
#[allow(deprecated)]
impl<'a> FusedIterator for LinesAny<'a> {}

/*
Section: Comparing strings
*/

/// Bytewise slice equality
/// NOTE: This function is (ab)used in rustc::middle::trans::_match
/// to compare &[u8] byte slices that are not necessarily valid UTF-8.
#[lang = "str_eq"]
#[inline]
fn eq_slice(a: &str, b: &str) -> bool {
    a.as_bytes() == b.as_bytes()
}

/*
Section: UTF-8 validation
*/

// use truncation to fit u64 into usize
const NONASCII_MASK: usize = 0x80808080_80808080u64 as usize;

/// Returns `true` if any byte in the word `x` is nonascii (>= 128).
#[inline]
fn contains_nonascii(x: usize) -> bool {
    (x & NONASCII_MASK) != 0
}

/// Walks through `iter` checking that it's a valid UTF-8 sequence,
/// returning `true` in that case, or, if it is invalid, `false` with
/// `iter` reset such that it is pointing at the first byte in the
/// invalid sequence.
#[inline(always)]
fn run_utf8_validation(v: &[u8]) -> Result<(), Utf8Error> {
    let mut index = 0;
    let len = v.len();

    let usize_bytes = mem::size_of::<usize>();
    let ascii_block_size = 2 * usize_bytes;
    let blocks_end = if len >= ascii_block_size { len - ascii_block_size + 1 } else { 0 };

    while index < len {
        let old_offset = index;
        macro_rules! err {
            ($error_len: expr) => {
                return Err(Utf8Error {
                    valid_up_to: old_offset,
                    error_len: $error_len,
                })
            }
        }

        macro_rules! next { () => {{
            index += 1;
            // we needed data, but there was none: error!
            if index >= len {
                err!(None)
            }
            v[index]
        }}}

        let first = v[index];
        if first >= 128 {
            let w = UTF8_CHAR_WIDTH[first as usize];
            // 2-byte encoding is for codepoints  \u{0080} to  \u{07ff}
            //        first  C2 80        last DF BF
            // 3-byte encoding is for codepoints  \u{0800} to  \u{ffff}
            //        first  E0 A0 80     last EF BF BF
            //   excluding surrogates codepoints  \u{d800} to  \u{dfff}
            //               ED A0 80 to       ED BF BF
            // 4-byte encoding is for codepoints \u{1000}0 to \u{10ff}ff
            //        first  F0 90 80 80  last F4 8F BF BF
            //
            // Use the UTF-8 syntax from the RFC
            //
            // https://tools.ietf.org/html/rfc3629
            // UTF8-1      = %x00-7F
            // UTF8-2      = %xC2-DF UTF8-tail
            // UTF8-3      = %xE0 %xA0-BF UTF8-tail / %xE1-EC 2( UTF8-tail ) /
            //               %xED %x80-9F UTF8-tail / %xEE-EF 2( UTF8-tail )
            // UTF8-4      = %xF0 %x90-BF 2( UTF8-tail ) / %xF1-F3 3( UTF8-tail ) /
            //               %xF4 %x80-8F 2( UTF8-tail )
            match w {
                2 => if next!() & !CONT_MASK != TAG_CONT_U8 {
                    err!(Some(1))
                },
                3 => {
                    match (first, next!()) {
                        (0xE0         , 0xA0 ... 0xBF) |
                        (0xE1 ... 0xEC, 0x80 ... 0xBF) |
                        (0xED         , 0x80 ... 0x9F) |
                        (0xEE ... 0xEF, 0x80 ... 0xBF) => {}
                        _ => err!(Some(1))
                    }
                    if next!() & !CONT_MASK != TAG_CONT_U8 {
                        err!(Some(2))
                    }
                }
                4 => {
                    match (first, next!()) {
                        (0xF0         , 0x90 ... 0xBF) |
                        (0xF1 ... 0xF3, 0x80 ... 0xBF) |
                        (0xF4         , 0x80 ... 0x8F) => {}
                        _ => err!(Some(1))
                    }
                    if next!() & !CONT_MASK != TAG_CONT_U8 {
                        err!(Some(2))
                    }
                    if next!() & !CONT_MASK != TAG_CONT_U8 {
                        err!(Some(3))
                    }
                }
                _ => err!(Some(1))
            }
            index += 1;
        } else {
            // Ascii case, try to skip forward quickly.
            // When the pointer is aligned, read 2 words of data per iteration
            // until we find a word containing a non-ascii byte.
            let ptr = v.as_ptr();
            let align = (ptr as usize + index) & (usize_bytes - 1);
            if align == 0 {
                while index < blocks_end {
                    unsafe {
                        let block = ptr.offset(index as isize) as *const usize;
                        // break if there is a nonascii byte
                        let zu = contains_nonascii(*block);
                        let zv = contains_nonascii(*block.offset(1));
                        if zu | zv {
                            break;
                        }
                    }
                    index += ascii_block_size;
                }
                // step from the point where the wordwise loop stopped
                while index < len && v[index] < 128 {
                    index += 1;
                }
            } else {
                index += 1;
            }
        }
    }

    Ok(())
}

// https://tools.ietf.org/html/rfc3629
static UTF8_CHAR_WIDTH: [u8; 256] = [
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 0x1F
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 0x3F
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 0x5F
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, // 0x7F
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 0x9F
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, // 0xBF
0,0,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2, // 0xDF
3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3, // 0xEF
4,4,4,4,4,0,0,0,0,0,0,0,0,0,0,0, // 0xFF
];

/// Given a first byte, determines how many bytes are in this UTF-8 character.
#[unstable(feature = "str_internals", issue = "0")]
#[inline]
pub fn utf8_char_width(b: u8) -> usize {
    return UTF8_CHAR_WIDTH[b as usize] as usize;
}

/// Mask of the value bits of a continuation byte.
const CONT_MASK: u8 = 0b0011_1111;
/// Value of the tag bits (tag mask is !CONT_MASK) of a continuation byte.
const TAG_CONT_U8: u8 = 0b1000_0000;

/*
Section: Trait implementations
*/

mod traits {
    use cmp::Ordering;
    use ops;
    use str::eq_slice;

    /// Implements ordering of strings.
    ///
    /// Strings are ordered  lexicographically by their byte values.  This orders Unicode code
    /// points based on their positions in the code charts.  This is not necessarily the same as
    /// "alphabetical" order, which varies by language and locale.  Sorting strings according to
    /// culturally-accepted standards requires locale-specific data that is outside the scope of
    /// the `str` type.
    #[stable(feature = "rust1", since = "1.0.0")]
    impl Ord for str {
        #[inline]
        fn cmp(&self, other: &str) -> Ordering {
            self.as_bytes().cmp(other.as_bytes())
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl PartialEq for str {
        #[inline]
        fn eq(&self, other: &str) -> bool {
            eq_slice(self, other)
        }
        #[inline]
        fn ne(&self, other: &str) -> bool { !(*self).eq(other) }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl Eq for str {}

    /// Implements comparison operations on strings.
    ///
    /// Strings are compared lexicographically by their byte values.  This compares Unicode code
    /// points based on their positions in the code charts.  This is not necessarily the same as
    /// "alphabetical" order, which varies by language and locale.  Comparing strings according to
    /// culturally-accepted standards requires locale-specific data that is outside the scope of
    /// the `str` type.
    #[stable(feature = "rust1", since = "1.0.0")]
    impl PartialOrd for str {
        #[inline]
        fn partial_cmp(&self, other: &str) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    /// Implements substring slicing with syntax `&self[begin .. end]`.
    ///
    /// Returns a slice of the given string from the byte range
    /// [`begin`..`end`).
    ///
    /// This operation is `O(1)`.
    ///
    /// # Panics
    ///
    /// Panics if `begin` or `end` does not point to the starting
    /// byte offset of a character (as defined by `is_char_boundary`).
    /// Requires that `begin <= end` and `end <= len` where `len` is the
    /// length of the string.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    /// assert_eq!(&s[0 .. 1], "L");
    ///
    /// assert_eq!(&s[1 .. 9], "öwe 老");
    ///
    /// // these will panic:
    /// // byte 2 lies within `ö`:
    /// // &s[2 ..3];
    ///
    /// // byte 8 lies within `老`
    /// // &s[1 .. 8];
    ///
    /// // byte 100 is outside the string
    /// // &s[3 .. 100];
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    impl ops::Index<ops::Range<usize>> for str {
        type Output = str;
        #[inline]
        fn index(&self, index: ops::Range<usize>) -> &str {
            // is_char_boundary checks that the index is in [0, .len()]
            if index.start <= index.end &&
               self.is_char_boundary(index.start) &&
               self.is_char_boundary(index.end) {
                unsafe { self.slice_unchecked(index.start, index.end) }
            } else {
                super::slice_error_fail(self, index.start, index.end)
            }
        }
    }

    /// Implements mutable substring slicing with syntax
    /// `&mut self[begin .. end]`.
    ///
    /// Returns a mutable slice of the given string from the byte range
    /// [`begin`..`end`).
    ///
    /// This operation is `O(1)`.
    ///
    /// # Panics
    ///
    /// Panics if `begin` or `end` does not point to the starting
    /// byte offset of a character (as defined by `is_char_boundary`).
    /// Requires that `begin <= end` and `end <= len` where `len` is the
    /// length of the string.
    #[stable(feature = "derefmut_for_string", since = "1.2.0")]
    impl ops::IndexMut<ops::Range<usize>> for str {
        #[inline]
        fn index_mut(&mut self, index: ops::Range<usize>) -> &mut str {
            // is_char_boundary checks that the index is in [0, .len()]
            if index.start <= index.end &&
               self.is_char_boundary(index.start) &&
               self.is_char_boundary(index.end) {
                unsafe { self.slice_mut_unchecked(index.start, index.end) }
            } else {
                super::slice_error_fail(self, index.start, index.end)
            }
        }
    }

    /// Implements substring slicing with syntax `&self[.. end]`.
    ///
    /// Returns a slice of the string from the beginning to byte offset
    /// `end`.
    ///
    /// Equivalent to `&self[0 .. end]`.
    #[stable(feature = "rust1", since = "1.0.0")]
    impl ops::Index<ops::RangeTo<usize>> for str {
        type Output = str;

        #[inline]
        fn index(&self, index: ops::RangeTo<usize>) -> &str {
            // is_char_boundary checks that the index is in [0, .len()]
            if self.is_char_boundary(index.end) {
                unsafe { self.slice_unchecked(0, index.end) }
            } else {
                super::slice_error_fail(self, 0, index.end)
            }
        }
    }

    /// Implements mutable substring slicing with syntax `&mut self[.. end]`.
    ///
    /// Returns a mutable slice of the string from the beginning to byte offset
    /// `end`.
    ///
    /// Equivalent to `&mut self[0 .. end]`.
    #[stable(feature = "derefmut_for_string", since = "1.2.0")]
    impl ops::IndexMut<ops::RangeTo<usize>> for str {
        #[inline]
        fn index_mut(&mut self, index: ops::RangeTo<usize>) -> &mut str {
            // is_char_boundary checks that the index is in [0, .len()]
            if self.is_char_boundary(index.end) {
                unsafe { self.slice_mut_unchecked(0, index.end) }
            } else {
                super::slice_error_fail(self, 0, index.end)
            }
        }
    }

    /// Implements substring slicing with syntax `&self[begin ..]`.
    ///
    /// Returns a slice of the string from byte offset `begin`
    /// to the end of the string.
    ///
    /// Equivalent to `&self[begin .. len]`.
    #[stable(feature = "rust1", since = "1.0.0")]
    impl ops::Index<ops::RangeFrom<usize>> for str {
        type Output = str;

        #[inline]
        fn index(&self, index: ops::RangeFrom<usize>) -> &str {
            // is_char_boundary checks that the index is in [0, .len()]
            if self.is_char_boundary(index.start) {
                unsafe { self.slice_unchecked(index.start, self.len()) }
            } else {
                super::slice_error_fail(self, index.start, self.len())
            }
        }
    }

    /// Implements mutable substring slicing with syntax `&mut self[begin ..]`.
    ///
    /// Returns a mutable slice of the string from byte offset `begin`
    /// to the end of the string.
    ///
    /// Equivalent to `&mut self[begin .. len]`.
    #[stable(feature = "derefmut_for_string", since = "1.2.0")]
    impl ops::IndexMut<ops::RangeFrom<usize>> for str {
        #[inline]
        fn index_mut(&mut self, index: ops::RangeFrom<usize>) -> &mut str {
            // is_char_boundary checks that the index is in [0, .len()]
            if self.is_char_boundary(index.start) {
                let len = self.len();
                unsafe { self.slice_mut_unchecked(index.start, len) }
            } else {
                super::slice_error_fail(self, index.start, self.len())
            }
        }
    }

    /// Implements substring slicing with syntax `&self[..]`.
    ///
    /// Returns a slice of the whole string. This operation can
    /// never panic.
    ///
    /// Equivalent to `&self[0 .. len]`.
    #[stable(feature = "rust1", since = "1.0.0")]
    impl ops::Index<ops::RangeFull> for str {
        type Output = str;

        #[inline]
        fn index(&self, _index: ops::RangeFull) -> &str {
            self
        }
    }

    /// Implements mutable substring slicing with syntax `&mut self[..]`.
    ///
    /// Returns a mutable slice of the whole string. This operation can
    /// never panic.
    ///
    /// Equivalent to `&mut self[0 .. len]`.
    #[stable(feature = "derefmut_for_string", since = "1.2.0")]
    impl ops::IndexMut<ops::RangeFull> for str {
        #[inline]
        fn index_mut(&mut self, _index: ops::RangeFull) -> &mut str {
            self
        }
    }

    #[unstable(feature = "inclusive_range",
               reason = "recently added, follows RFC",
               issue = "28237")]
    impl ops::Index<ops::RangeInclusive<usize>> for str {
        type Output = str;

        #[inline]
        fn index(&self, index: ops::RangeInclusive<usize>) -> &str {
            match index {
                ops::RangeInclusive::Empty { .. } => "",
                ops::RangeInclusive::NonEmpty { end, .. } if end == usize::max_value() =>
                    panic!("attempted to index slice up to maximum usize"),
                ops::RangeInclusive::NonEmpty { start, end } =>
                    self.index(start .. end+1)
            }
        }
    }
    #[unstable(feature = "inclusive_range",
               reason = "recently added, follows RFC",
               issue = "28237")]
    impl ops::Index<ops::RangeToInclusive<usize>> for str {
        type Output = str;

        #[inline]
        fn index(&self, index: ops::RangeToInclusive<usize>) -> &str {
            self.index(0...index.end)
        }
    }

    #[unstable(feature = "inclusive_range",
               reason = "recently added, follows RFC",
               issue = "28237")]
    impl ops::IndexMut<ops::RangeInclusive<usize>> for str {
        #[inline]
        fn index_mut(&mut self, index: ops::RangeInclusive<usize>) -> &mut str {
            match index {
                ops::RangeInclusive::Empty { .. } => &mut self[0..0], // `&mut ""` doesn't work
                ops::RangeInclusive::NonEmpty { end, .. } if end == usize::max_value() =>
                    panic!("attempted to index str up to maximum usize"),
                    ops::RangeInclusive::NonEmpty { start, end } =>
                        self.index_mut(start .. end+1)
            }
        }
    }
    #[unstable(feature = "inclusive_range",
               reason = "recently added, follows RFC",
               issue = "28237")]
    impl ops::IndexMut<ops::RangeToInclusive<usize>> for str {
        #[inline]
        fn index_mut(&mut self, index: ops::RangeToInclusive<usize>) -> &mut str {
            self.index_mut(0...index.end)
        }
    }
}

/// Methods for string slices
#[allow(missing_docs)]
#[doc(hidden)]
#[unstable(feature = "core_str_ext",
           reason = "stable interface provided by `impl str` in later crates",
           issue = "32110")]
pub trait StrExt {
    // NB there are no docs here are they're all located on the StrExt trait in
    // libcollections, not here.

    #[stable(feature = "core", since = "1.6.0")]
    fn contains<'a, P: Pattern<'a>>(&'a self, pat: P) -> bool;
    #[stable(feature = "core", since = "1.6.0")]
    fn chars(&self) -> Chars;
    #[stable(feature = "core", since = "1.6.0")]
    fn bytes(&self) -> Bytes;
    #[stable(feature = "core", since = "1.6.0")]
    fn char_indices(&self) -> CharIndices;
    #[stable(feature = "core", since = "1.6.0")]
    fn split<'a, P: Pattern<'a>>(&'a self, pat: P) -> Split<'a, P>;
    #[stable(feature = "core", since = "1.6.0")]
    fn rsplit<'a, P: Pattern<'a>>(&'a self, pat: P) -> RSplit<'a, P>
        where P::Searcher: ReverseSearcher<'a>;
    #[stable(feature = "core", since = "1.6.0")]
    fn splitn<'a, P: Pattern<'a>>(&'a self, count: usize, pat: P) -> SplitN<'a, P>;
    #[stable(feature = "core", since = "1.6.0")]
    fn rsplitn<'a, P: Pattern<'a>>(&'a self, count: usize, pat: P) -> RSplitN<'a, P>
        where P::Searcher: ReverseSearcher<'a>;
    #[stable(feature = "core", since = "1.6.0")]
    fn split_terminator<'a, P: Pattern<'a>>(&'a self, pat: P) -> SplitTerminator<'a, P>;
    #[stable(feature = "core", since = "1.6.0")]
    fn rsplit_terminator<'a, P: Pattern<'a>>(&'a self, pat: P) -> RSplitTerminator<'a, P>
        where P::Searcher: ReverseSearcher<'a>;
    #[stable(feature = "core", since = "1.6.0")]
    fn matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> Matches<'a, P>;
    #[stable(feature = "core", since = "1.6.0")]
    fn rmatches<'a, P: Pattern<'a>>(&'a self, pat: P) -> RMatches<'a, P>
        where P::Searcher: ReverseSearcher<'a>;
    #[stable(feature = "core", since = "1.6.0")]
    fn match_indices<'a, P: Pattern<'a>>(&'a self, pat: P) -> MatchIndices<'a, P>;
    #[stable(feature = "core", since = "1.6.0")]
    fn rmatch_indices<'a, P: Pattern<'a>>(&'a self, pat: P) -> RMatchIndices<'a, P>
        where P::Searcher: ReverseSearcher<'a>;
    #[stable(feature = "core", since = "1.6.0")]
    fn lines(&self) -> Lines;
    #[stable(feature = "core", since = "1.6.0")]
    #[rustc_deprecated(since = "1.6.0", reason = "use lines() instead now")]
    #[allow(deprecated)]
    fn lines_any(&self) -> LinesAny;
    #[stable(feature = "core", since = "1.6.0")]
    unsafe fn slice_unchecked(&self, begin: usize, end: usize) -> &str;
    #[stable(feature = "core", since = "1.6.0")]
    unsafe fn slice_mut_unchecked(&mut self, begin: usize, end: usize) -> &mut str;
    #[stable(feature = "core", since = "1.6.0")]
    fn starts_with<'a, P: Pattern<'a>>(&'a self, pat: P) -> bool;
    #[stable(feature = "core", since = "1.6.0")]
    fn ends_with<'a, P: Pattern<'a>>(&'a self, pat: P) -> bool
        where P::Searcher: ReverseSearcher<'a>;
    #[stable(feature = "core", since = "1.6.0")]
    fn trim_matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> &'a str
        where P::Searcher: DoubleEndedSearcher<'a>;
    #[stable(feature = "core", since = "1.6.0")]
    fn trim_left_matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> &'a str;
    #[stable(feature = "core", since = "1.6.0")]
    fn trim_right_matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> &'a str
        where P::Searcher: ReverseSearcher<'a>;
    #[stable(feature = "is_char_boundary", since = "1.9.0")]
    fn is_char_boundary(&self, index: usize) -> bool;
    #[stable(feature = "core", since = "1.6.0")]
    fn as_bytes(&self) -> &[u8];
    #[stable(feature = "core", since = "1.6.0")]
    fn find<'a, P: Pattern<'a>>(&'a self, pat: P) -> Option<usize>;
    #[stable(feature = "core", since = "1.6.0")]
    fn rfind<'a, P: Pattern<'a>>(&'a self, pat: P) -> Option<usize>
        where P::Searcher: ReverseSearcher<'a>;
    fn find_str<'a, P: Pattern<'a>>(&'a self, pat: P) -> Option<usize>;
    #[stable(feature = "core", since = "1.6.0")]
    fn split_at(&self, mid: usize) -> (&str, &str);
    #[stable(feature = "core", since = "1.6.0")]
    fn split_at_mut(&mut self, mid: usize) -> (&mut str, &mut str);
    #[stable(feature = "core", since = "1.6.0")]
    fn as_ptr(&self) -> *const u8;
    #[stable(feature = "core", since = "1.6.0")]
    fn len(&self) -> usize;
    #[stable(feature = "core", since = "1.6.0")]
    fn is_empty(&self) -> bool;
    #[stable(feature = "core", since = "1.6.0")]
    fn parse<'a, T: TryFrom<&'a str>>(&'a self) -> Result<T, T::Error>;
}

// truncate `&str` to length at most equal to `max`
// return `true` if it were truncated, and the new str.
fn truncate_to_char_boundary(s: &str, mut max: usize) -> (bool, &str) {
    if max >= s.len() {
        (false, s)
    } else {
        while !s.is_char_boundary(max) {
            max -= 1;
        }
        (true, &s[..max])
    }
}

#[inline(never)]
#[cold]
fn slice_error_fail(s: &str, begin: usize, end: usize) -> ! {
    const MAX_DISPLAY_LENGTH: usize = 256;
    let (truncated, s_trunc) = truncate_to_char_boundary(s, MAX_DISPLAY_LENGTH);
    let ellipsis = if truncated { "[...]" } else { "" };

    // 1. out of bounds
    if begin > s.len() || end > s.len() {
        let oob_index = if begin > s.len() { begin } else { end };
        panic!("byte index {} is out of bounds of `{}`{}", oob_index, s_trunc, ellipsis);
    }

    // 2. begin <= end
    assert!(begin <= end, "begin <= end ({} <= {}) when slicing `{}`{}",
            begin, end, s_trunc, ellipsis);

    // 3. character boundary
    let index = if !s.is_char_boundary(begin) { begin } else { end };
    // find the character
    let mut char_start = index;
    while !s.is_char_boundary(char_start) {
        char_start -= 1;
    }
    // `char_start` must be less than len and a char boundary
    let ch = s[char_start..].chars().next().unwrap();
    let char_range = char_start .. char_start + ch.len_utf8();
    panic!("byte index {} is not a char boundary; it is inside {:?} (bytes {:?}) of `{}`{}",
           index, ch, char_range, s_trunc, ellipsis);
}

#[stable(feature = "core", since = "1.6.0")]
impl StrExt for str {
    #[inline]
    fn contains<'a, P: Pattern<'a>>(&'a self, pat: P) -> bool {
        pat.is_contained_in(self)
    }

    #[inline]
    fn chars(&self) -> Chars {
        Chars{iter: self.as_bytes().iter()}
    }

    #[inline]
    fn bytes(&self) -> Bytes {
        Bytes(self.as_bytes().iter().cloned())
    }

    #[inline]
    fn char_indices(&self) -> CharIndices {
        CharIndices { front_offset: 0, iter: self.chars() }
    }

    #[inline]
    fn split<'a, P: Pattern<'a>>(&'a self, pat: P) -> Split<'a, P> {
        Split(SplitInternal {
            start: 0,
            end: self.len(),
            matcher: pat.into_searcher(self),
            allow_trailing_empty: true,
            finished: false,
        })
    }

    #[inline]
    fn rsplit<'a, P: Pattern<'a>>(&'a self, pat: P) -> RSplit<'a, P>
        where P::Searcher: ReverseSearcher<'a>
    {
        RSplit(self.split(pat).0)
    }

    #[inline]
    fn splitn<'a, P: Pattern<'a>>(&'a self, count: usize, pat: P) -> SplitN<'a, P> {
        SplitN(SplitNInternal {
            iter: self.split(pat).0,
            count: count,
        })
    }

    #[inline]
    fn rsplitn<'a, P: Pattern<'a>>(&'a self, count: usize, pat: P) -> RSplitN<'a, P>
        where P::Searcher: ReverseSearcher<'a>
    {
        RSplitN(self.splitn(count, pat).0)
    }

    #[inline]
    fn split_terminator<'a, P: Pattern<'a>>(&'a self, pat: P) -> SplitTerminator<'a, P> {
        SplitTerminator(SplitInternal {
            allow_trailing_empty: false,
            ..self.split(pat).0
        })
    }

    #[inline]
    fn rsplit_terminator<'a, P: Pattern<'a>>(&'a self, pat: P) -> RSplitTerminator<'a, P>
        where P::Searcher: ReverseSearcher<'a>
    {
        RSplitTerminator(self.split_terminator(pat).0)
    }

    #[inline]
    fn matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> Matches<'a, P> {
        Matches(MatchesInternal(pat.into_searcher(self)))
    }

    #[inline]
    fn rmatches<'a, P: Pattern<'a>>(&'a self, pat: P) -> RMatches<'a, P>
        where P::Searcher: ReverseSearcher<'a>
    {
        RMatches(self.matches(pat).0)
    }

    #[inline]
    fn match_indices<'a, P: Pattern<'a>>(&'a self, pat: P) -> MatchIndices<'a, P> {
        MatchIndices(MatchIndicesInternal(pat.into_searcher(self)))
    }

    #[inline]
    fn rmatch_indices<'a, P: Pattern<'a>>(&'a self, pat: P) -> RMatchIndices<'a, P>
        where P::Searcher: ReverseSearcher<'a>
    {
        RMatchIndices(self.match_indices(pat).0)
    }
    #[inline]
    fn lines(&self) -> Lines {
        Lines(self.split_terminator('\n').map(LinesAnyMap))
    }

    #[inline]
    #[allow(deprecated)]
    fn lines_any(&self) -> LinesAny {
        LinesAny(self.lines())
    }

    #[inline]
    unsafe fn slice_unchecked(&self, begin: usize, end: usize) -> &str {
        let ptr = self.as_ptr().offset(begin as isize);
        let len = end - begin;
        from_utf8_unchecked(slice::from_raw_parts(ptr, len))
    }

    #[inline]
    unsafe fn slice_mut_unchecked(&mut self, begin: usize, end: usize) -> &mut str {
        let ptr = self.as_ptr().offset(begin as isize);
        let len = end - begin;
        mem::transmute(slice::from_raw_parts_mut(ptr as *mut u8, len))
    }

    #[inline]
    fn starts_with<'a, P: Pattern<'a>>(&'a self, pat: P) -> bool {
        pat.is_prefix_of(self)
    }

    #[inline]
    fn ends_with<'a, P: Pattern<'a>>(&'a self, pat: P) -> bool
        where P::Searcher: ReverseSearcher<'a>
    {
        pat.is_suffix_of(self)
    }

    #[inline]
    fn trim_matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> &'a str
        where P::Searcher: DoubleEndedSearcher<'a>
    {
        let mut i = 0;
        let mut j = 0;
        let mut matcher = pat.into_searcher(self);
        if let Some((a, b)) = matcher.next_reject() {
            i = a;
            j = b; // Remember earliest known match, correct it below if
                   // last match is different
        }
        if let Some((_, b)) = matcher.next_reject_back() {
            j = b;
        }
        unsafe {
            // Searcher is known to return valid indices
            self.slice_unchecked(i, j)
        }
    }

    #[inline]
    fn trim_left_matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> &'a str {
        let mut i = self.len();
        let mut matcher = pat.into_searcher(self);
        if let Some((a, _)) = matcher.next_reject() {
            i = a;
        }
        unsafe {
            // Searcher is known to return valid indices
            self.slice_unchecked(i, self.len())
        }
    }

    #[inline]
    fn trim_right_matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> &'a str
        where P::Searcher: ReverseSearcher<'a>
    {
        let mut j = 0;
        let mut matcher = pat.into_searcher(self);
        if let Some((_, b)) = matcher.next_reject_back() {
            j = b;
        }
        unsafe {
            // Searcher is known to return valid indices
            self.slice_unchecked(0, j)
        }
    }

    #[inline]
    fn is_char_boundary(&self, index: usize) -> bool {
        // 0 and len are always ok.
        // Test for 0 explicitly so that it can optimize out the check
        // easily and skip reading string data for that case.
        if index == 0 || index == self.len() { return true; }
        match self.as_bytes().get(index) {
            None => false,
            // This is bit magic equivalent to: b < 128 || b >= 192
            Some(&b) => (b as i8) >= -0x40,
        }
    }

    #[inline]
    fn as_bytes(&self) -> &[u8] {
        unsafe { mem::transmute(self) }
    }

    fn find<'a, P: Pattern<'a>>(&'a self, pat: P) -> Option<usize> {
        pat.into_searcher(self).next_match().map(|(i, _)| i)
    }

    fn rfind<'a, P: Pattern<'a>>(&'a self, pat: P) -> Option<usize>
        where P::Searcher: ReverseSearcher<'a>
    {
        pat.into_searcher(self).next_match_back().map(|(i, _)| i)
    }

    fn find_str<'a, P: Pattern<'a>>(&'a self, pat: P) -> Option<usize> {
        self.find(pat)
    }

    #[inline]
    fn split_at(&self, mid: usize) -> (&str, &str) {
        // is_char_boundary checks that the index is in [0, .len()]
        if self.is_char_boundary(mid) {
            unsafe {
                (self.slice_unchecked(0, mid),
                 self.slice_unchecked(mid, self.len()))
            }
        } else {
            slice_error_fail(self, 0, mid)
        }
    }

    fn split_at_mut(&mut self, mid: usize) -> (&mut str, &mut str) {
        // is_char_boundary checks that the index is in [0, .len()]
        if self.is_char_boundary(mid) {
            let len = self.len();
            let ptr = self.as_ptr() as *mut u8;
            unsafe {
                (from_raw_parts_mut(ptr, mid),
                 from_raw_parts_mut(ptr.offset(mid as isize), len - mid))
            }
        } else {
            slice_error_fail(self, 0, mid)
        }
    }

    #[inline]
    fn as_ptr(&self) -> *const u8 {
        self as *const str as *const u8
    }

    #[inline]
    fn len(&self) -> usize {
        self.as_bytes().len()
    }

    #[inline]
    fn is_empty(&self) -> bool { self.len() == 0 }

    #[inline]
    fn parse<'a, T>(&'a self) -> Result<T, T::Error> where T: TryFrom<&'a str> {
        T::try_from(self)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl AsRef<[u8]> for str {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_bytes()
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl<'a> Default for &'a str {
    /// Creates an empty str
    fn default() -> &'a str { "" }
}
