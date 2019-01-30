//! String manipulation
//!
//! For more details, see std::str

#![stable(feature = "rust1", since = "1.0.0")]

use self::pattern::Pattern;
use self::pattern::{Searcher, ReverseSearcher, DoubleEndedSearcher};

use char;
use fmt;
use iter::{Map, Cloned, FusedIterator, TrustedLen, TrustedRandomAccess, Filter};
use slice::{self, SliceIndex, Split as SliceSplit};
use mem;

pub mod pattern;

#[unstable(feature = "str_internals", issue = "0")]
#[allow(missing_docs)]
pub mod lossy;

/// Parse a value from a string
///
/// `FromStr`'s [`from_str`] method is often used implicitly, through
/// [`str`]'s [`parse`] method. See [`parse`]'s documentation for examples.
///
/// [`from_str`]: #tymethod.from_str
/// [`str`]: ../../std/primitive.str.html
/// [`parse`]: ../../std/primitive.str.html#method.parse
///
/// `FromStr` does not have a lifetime parameter, and so you can only parse types
/// that do not contain a lifetime parameter themselves. In other words, you can
/// parse an `i32` with `FromStr`, but not a `&i32`. You can parse a struct that
/// contains an `i32`, but not one that contains an `&i32`.
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
///                                  .split(',')
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
    /// If parsing succeeds, return the value inside [`Ok`], otherwise
    /// when the string is ill-formatted return an error specific to the
    /// inside [`Err`]. The error type is specific to implementation of the trait.
    ///
    /// [`Ok`]: ../../std/result/enum.Result.html#variant.Ok
    /// [`Err`]: ../../std/result/enum.Result.html#variant.Err
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

/// Errors which can occur when attempting to interpret a sequence of [`u8`]
/// as a string.
///
/// [`u8`]: ../../std/primitive.u8.html
///
/// As such, the `from_utf8` family of functions and methods for both [`String`]s
/// and [`&str`]s make use of this error, for example.
///
/// [`String`]: ../../std/string/struct.String.html#method.from_utf8
/// [`&str`]: ../../std/str/fn.from_utf8.html
///
/// # Examples
///
/// This error type’s methods can be used to create functionality
/// similar to `String::from_utf8_lossy` without allocating heap memory:
///
/// ```
/// fn from_utf8_lossy<F>(mut input: &[u8], mut push: F) where F: FnMut(&str) {
///     loop {
///         match ::std::str::from_utf8(input) {
///             Ok(valid) => {
///                 push(valid);
///                 break
///             }
///             Err(error) => {
///                 let (valid, after_valid) = input.split_at(error.valid_up_to());
///                 unsafe {
///                     push(::std::str::from_utf8_unchecked(valid))
///                 }
///                 push("\u{FFFD}");
///
///                 if let Some(invalid_sequence_length) = error.error_len() {
///                     input = &after_valid[invalid_sequence_length..]
///                 } else {
///                     break
///                 }
///             }
///         }
///     }
/// }
/// ```
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
    ///   (after inserting a [`U+FFFD REPLACEMENT CHARACTER`][U+FFFD]) in case of
    ///   lossy decoding.
    ///
    /// [U+FFFD]: ../../std/char/constant.REPLACEMENT_CHARACTER.html
    #[stable(feature = "utf8_error_error_len", since = "1.20.0")]
    pub fn error_len(&self) -> Option<usize> {
        self.error_len.map(|len| len as usize)
    }
}

/// Converts a slice of bytes to a string slice.
///
/// A string slice ([`&str`]) is made of bytes ([`u8`]), and a byte slice
/// ([`&[u8]`][byteslice]) is made of bytes, so this function converts between
/// the two. Not all byte slices are valid string slices, however: [`&str`] requires
/// that it is valid UTF-8. `from_utf8()` checks to ensure that the bytes are valid
/// UTF-8, and then does the conversion.
///
/// [`&str`]: ../../std/primitive.str.html
/// [`u8`]: ../../std/primitive.u8.html
/// [byteslice]: ../../std/primitive.slice.html
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
/// Because you can stack-allocate a `[u8; N]`, and you can take a
/// [`&[u8]`][byteslice] of it, this function is one way to have a
/// stack-allocated string. There is an example of this in the
/// examples section below.
///
/// [byteslice]: ../../std/primitive.slice.html
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
/// See the docs for [`Utf8Error`][error] for more details on the kinds of
/// errors that can be returned.
///
/// [error]: struct.Utf8Error.html
#[stable(feature = "str_mut_extras", since = "1.20.0")]
pub fn from_utf8_mut(v: &mut [u8]) -> Result<&mut str, Utf8Error> {
    run_utf8_validation(v)?;
    Ok(unsafe { from_utf8_unchecked_mut(v) })
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
#[inline]
#[stable(feature = "rust1", since = "1.0.0")]
pub unsafe fn from_utf8_unchecked(v: &[u8]) -> &str {
    &*(v as *const [u8] as *const str)
}

/// Converts a slice of bytes to a string slice without checking
/// that the string contains valid UTF-8; mutable version.
///
/// See the immutable version, [`from_utf8_unchecked()`][fromutf8], for more information.
///
/// [fromutf8]: fn.from_utf8_unchecked.html
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
    &mut *(v as *mut [u8] as *mut str)
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

/// An iterator over the [`char`]s of a string slice.
///
/// [`char`]: ../../std/primitive.char.html
///
/// This struct is created by the [`chars`] method on [`str`].
/// See its documentation for more.
///
/// [`chars`]: ../../std/primitive.str.html#method.chars
/// [`str`]: ../../std/primitive.str.html
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

/// Checks whether the byte is a UTF-8 continuation byte (i.e., starts with the
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
    let x = *bytes.next()?;
    if x < 128 {
        return Some(x as u32)
    }

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
    let w = match *bytes.next_back()? {
        next_byte if next_byte < 128 => return Some(next_byte as u32),
        back_byte => back_byte,
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

#[stable(feature = "fused", since = "1.26.0")]
impl FusedIterator for Chars<'_> {}

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

/// An iterator over the [`char`]s of a string slice, and their positions.
///
/// [`char`]: ../../std/primitive.char.html
///
/// This struct is created by the [`char_indices`] method on [`str`].
/// See its documentation for more.
///
/// [`char_indices`]: ../../std/primitive.str.html#method.char_indices
/// [`str`]: ../../std/primitive.str.html
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
        self.iter.next_back().map(|ch| {
            let index = self.front_offset + self.iter.iter.len();
            (index, ch)
        })
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl FusedIterator for CharIndices<'_> {}

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

/// An iterator over the bytes of a string slice.
///
/// This struct is created by the [`bytes`] method on [`str`].
/// See its documentation for more.
///
/// [`bytes`]: ../../std/primitive.str.html#method.bytes
/// [`str`]: ../../std/primitive.str.html
#[stable(feature = "rust1", since = "1.0.0")]
#[derive(Clone, Debug)]
pub struct Bytes<'a>(Cloned<slice::Iter<'a, u8>>);

#[stable(feature = "rust1", since = "1.0.0")]
impl Iterator for Bytes<'_> {
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

    #[inline]
    fn all<F>(&mut self, f: F) -> bool where F: FnMut(Self::Item) -> bool {
        self.0.all(f)
    }

    #[inline]
    fn any<F>(&mut self, f: F) -> bool where F: FnMut(Self::Item) -> bool {
        self.0.any(f)
    }

    #[inline]
    fn find<P>(&mut self, predicate: P) -> Option<Self::Item> where
        P: FnMut(&Self::Item) -> bool
    {
        self.0.find(predicate)
    }

    #[inline]
    fn position<P>(&mut self, predicate: P) -> Option<usize> where
        P: FnMut(Self::Item) -> bool
    {
        self.0.position(predicate)
    }

    #[inline]
    fn rposition<P>(&mut self, predicate: P) -> Option<usize> where
        P: FnMut(Self::Item) -> bool
    {
        self.0.rposition(predicate)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl DoubleEndedIterator for Bytes<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<u8> {
        self.0.next_back()
    }

    #[inline]
    fn rfind<P>(&mut self, predicate: P) -> Option<Self::Item> where
        P: FnMut(&Self::Item) -> bool
    {
        self.0.rfind(predicate)
    }
}

#[stable(feature = "rust1", since = "1.0.0")]
impl ExactSizeIterator for Bytes<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl FusedIterator for Bytes<'_> {}

#[unstable(feature = "trusted_len", issue = "37572")]
unsafe impl TrustedLen for Bytes<'_> {}

#[doc(hidden)]
unsafe impl<'a> TrustedRandomAccess for Bytes<'a> {
    unsafe fn get_unchecked(&mut self, i: usize) -> u8 {
        self.0.get_unchecked(i)
    }
    fn may_have_side_effect() -> bool { false }
}

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

        // Kind of delegation - either single ended or double ended
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

        #[stable(feature = "fused", since = "1.26.0")]
        impl<'a, P: Pattern<'a>> FusedIterator for $forward_iterator<'a, P> {}

        #[stable(feature = "fused", since = "1.26.0")]
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
                let string = self.matcher.haystack().get_unchecked(self.start..self.end);
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
                let elt = haystack.get_unchecked(self.start..a);
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
                let elt = haystack.get_unchecked(b..self.end);
                self.end = a;
                Some(elt)
            },
            None => unsafe {
                self.finished = true;
                Some(haystack.get_unchecked(self.start..self.end))
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
            (start, self.0.haystack().get_unchecked(start..end))
        })
    }

    #[inline]
    fn next_back(&mut self) -> Option<(usize, &'a str)>
        where P::Searcher: ReverseSearcher<'a>
    {
        self.0.next_match_back().map(|(start, end)| unsafe {
            (start, self.0.haystack().get_unchecked(start..end))
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
            self.0.haystack().get_unchecked(a..b)
        })
    }

    #[inline]
    fn next_back(&mut self) -> Option<&'a str>
        where P::Searcher: ReverseSearcher<'a>
    {
        self.0.next_match_back().map(|(a, b)| unsafe {
            // Indices are known to be on utf8 boundaries
            self.0.haystack().get_unchecked(a..b)
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

/// An iterator over the lines of a string, as string slices.
///
/// This struct is created with the [`lines`] method on [`str`].
/// See its documentation for more.
///
/// [`lines`]: ../../std/primitive.str.html#method.lines
/// [`str`]: ../../std/primitive.str.html
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

#[stable(feature = "fused", since = "1.26.0")]
impl FusedIterator for Lines<'_> {}

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

#[stable(feature = "fused", since = "1.26.0")]
#[allow(deprecated)]
impl FusedIterator for LinesAny<'_> {}

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

/// Walks through `v` checking that it's a valid UTF-8 sequence,
/// returning `Ok(())` in that case, or, if it is invalid, `Err(err)`.
#[inline]
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
                        (0xE0         , 0xA0 ..= 0xBF) |
                        (0xE1 ..= 0xEC, 0x80 ..= 0xBF) |
                        (0xED         , 0x80 ..= 0x9F) |
                        (0xEE ..= 0xEF, 0x80 ..= 0xBF) => {}
                        _ => err!(Some(1))
                    }
                    if next!() & !CONT_MASK != TAG_CONT_U8 {
                        err!(Some(2))
                    }
                }
                4 => {
                    match (first, next!()) {
                        (0xF0         , 0x90 ..= 0xBF) |
                        (0xF1 ..= 0xF3, 0x80 ..= 0xBF) |
                        (0xF4         , 0x80 ..= 0x8F) => {}
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
            let align = unsafe {
                // the offset is safe, because `index` is guaranteed inbounds
                ptr.add(index).align_offset(usize_bytes)
            };
            if align == 0 {
                while index < blocks_end {
                    unsafe {
                        let block = ptr.add(index) as *const usize;
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
    UTF8_CHAR_WIDTH[b as usize] as usize
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
    use slice::{self, SliceIndex};

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
            self.as_bytes() == other.as_bytes()
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

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<I> ops::Index<I> for str
    where
        I: SliceIndex<str>,
    {
        type Output = I::Output;

        #[inline]
        fn index(&self, index: I) -> &I::Output {
            index.index(self)
        }
    }

    #[stable(feature = "rust1", since = "1.0.0")]
    impl<I> ops::IndexMut<I> for str
    where
        I: SliceIndex<str>,
    {
        #[inline]
        fn index_mut(&mut self, index: I) -> &mut I::Output {
            index.index_mut(self)
        }
    }

    #[inline(never)]
    #[cold]
    fn str_index_overflow_fail() -> ! {
        panic!("attempted to index str up to maximum usize");
    }

    /// Implements substring slicing with syntax `&self[..]` or `&mut self[..]`.
    ///
    /// Returns a slice of the whole string, i.e., returns `&self` or `&mut
    /// self`. Equivalent to `&self[0 .. len]` or `&mut self[0 .. len]`. Unlike
    /// other indexing operations, this can never panic.
    ///
    /// This operation is `O(1)`.
    ///
    /// Prior to 1.20.0, these indexing operations were still supported by
    /// direct implementation of `Index` and `IndexMut`.
    ///
    /// Equivalent to `&self[0 .. len]` or `&mut self[0 .. len]`.
    #[stable(feature = "str_checked_slicing", since = "1.20.0")]
    impl SliceIndex<str> for ops::RangeFull {
        type Output = str;
        #[inline]
        fn get(self, slice: &str) -> Option<&Self::Output> {
            Some(slice)
        }
        #[inline]
        fn get_mut(self, slice: &mut str) -> Option<&mut Self::Output> {
            Some(slice)
        }
        #[inline]
        unsafe fn get_unchecked(self, slice: &str) -> &Self::Output {
            slice
        }
        #[inline]
        unsafe fn get_unchecked_mut(self, slice: &mut str) -> &mut Self::Output {
            slice
        }
        #[inline]
        fn index(self, slice: &str) -> &Self::Output {
            slice
        }
        #[inline]
        fn index_mut(self, slice: &mut str) -> &mut Self::Output {
            slice
        }
    }

    /// Implements substring slicing with syntax `&self[begin .. end]` or `&mut
    /// self[begin .. end]`.
    ///
    /// Returns a slice of the given string from the byte range
    /// [`begin`, `end`).
    ///
    /// This operation is `O(1)`.
    ///
    /// Prior to 1.20.0, these indexing operations were still supported by
    /// direct implementation of `Index` and `IndexMut`.
    ///
    /// # Panics
    ///
    /// Panics if `begin` or `end` does not point to the starting byte offset of
    /// a character (as defined by `is_char_boundary`), if `begin > end`, or if
    /// `end > len`.
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
    #[stable(feature = "str_checked_slicing", since = "1.20.0")]
    impl SliceIndex<str> for ops::Range<usize> {
        type Output = str;
        #[inline]
        fn get(self, slice: &str) -> Option<&Self::Output> {
            if self.start <= self.end &&
               slice.is_char_boundary(self.start) &&
               slice.is_char_boundary(self.end) {
                Some(unsafe { self.get_unchecked(slice) })
            } else {
                None
            }
        }
        #[inline]
        fn get_mut(self, slice: &mut str) -> Option<&mut Self::Output> {
            if self.start <= self.end &&
               slice.is_char_boundary(self.start) &&
               slice.is_char_boundary(self.end) {
                Some(unsafe { self.get_unchecked_mut(slice) })
            } else {
                None
            }
        }
        #[inline]
        unsafe fn get_unchecked(self, slice: &str) -> &Self::Output {
            let ptr = slice.as_ptr().add(self.start);
            let len = self.end - self.start;
            super::from_utf8_unchecked(slice::from_raw_parts(ptr, len))
        }
        #[inline]
        unsafe fn get_unchecked_mut(self, slice: &mut str) -> &mut Self::Output {
            let ptr = slice.as_ptr().add(self.start);
            let len = self.end - self.start;
            super::from_utf8_unchecked_mut(slice::from_raw_parts_mut(ptr as *mut u8, len))
        }
        #[inline]
        fn index(self, slice: &str) -> &Self::Output {
            let (start, end) = (self.start, self.end);
            self.get(slice).unwrap_or_else(|| super::slice_error_fail(slice, start, end))
        }
        #[inline]
        fn index_mut(self, slice: &mut str) -> &mut Self::Output {
            // is_char_boundary checks that the index is in [0, .len()]
            // cannot reuse `get` as above, because of NLL trouble
            if self.start <= self.end &&
               slice.is_char_boundary(self.start) &&
               slice.is_char_boundary(self.end) {
                unsafe { self.get_unchecked_mut(slice) }
            } else {
                super::slice_error_fail(slice, self.start, self.end)
            }
        }
    }

    /// Implements substring slicing with syntax `&self[.. end]` or `&mut
    /// self[.. end]`.
    ///
    /// Returns a slice of the given string from the byte range [`0`, `end`).
    /// Equivalent to `&self[0 .. end]` or `&mut self[0 .. end]`.
    ///
    /// This operation is `O(1)`.
    ///
    /// Prior to 1.20.0, these indexing operations were still supported by
    /// direct implementation of `Index` and `IndexMut`.
    ///
    /// # Panics
    ///
    /// Panics if `end` does not point to the starting byte offset of a
    /// character (as defined by `is_char_boundary`), or if `end > len`.
    #[stable(feature = "str_checked_slicing", since = "1.20.0")]
    impl SliceIndex<str> for ops::RangeTo<usize> {
        type Output = str;
        #[inline]
        fn get(self, slice: &str) -> Option<&Self::Output> {
            if slice.is_char_boundary(self.end) {
                Some(unsafe { self.get_unchecked(slice) })
            } else {
                None
            }
        }
        #[inline]
        fn get_mut(self, slice: &mut str) -> Option<&mut Self::Output> {
            if slice.is_char_boundary(self.end) {
                Some(unsafe { self.get_unchecked_mut(slice) })
            } else {
                None
            }
        }
        #[inline]
        unsafe fn get_unchecked(self, slice: &str) -> &Self::Output {
            let ptr = slice.as_ptr();
            super::from_utf8_unchecked(slice::from_raw_parts(ptr, self.end))
        }
        #[inline]
        unsafe fn get_unchecked_mut(self, slice: &mut str) -> &mut Self::Output {
            let ptr = slice.as_ptr();
            super::from_utf8_unchecked_mut(slice::from_raw_parts_mut(ptr as *mut u8, self.end))
        }
        #[inline]
        fn index(self, slice: &str) -> &Self::Output {
            let end = self.end;
            self.get(slice).unwrap_or_else(|| super::slice_error_fail(slice, 0, end))
        }
        #[inline]
        fn index_mut(self, slice: &mut str) -> &mut Self::Output {
            // is_char_boundary checks that the index is in [0, .len()]
            if slice.is_char_boundary(self.end) {
                unsafe { self.get_unchecked_mut(slice) }
            } else {
                super::slice_error_fail(slice, 0, self.end)
            }
        }
    }

    /// Implements substring slicing with syntax `&self[begin ..]` or `&mut
    /// self[begin ..]`.
    ///
    /// Returns a slice of the given string from the byte range [`begin`,
    /// `len`). Equivalent to `&self[begin .. len]` or `&mut self[begin ..
    /// len]`.
    ///
    /// This operation is `O(1)`.
    ///
    /// Prior to 1.20.0, these indexing operations were still supported by
    /// direct implementation of `Index` and `IndexMut`.
    ///
    /// # Panics
    ///
    /// Panics if `begin` does not point to the starting byte offset of
    /// a character (as defined by `is_char_boundary`), or if `begin >= len`.
    #[stable(feature = "str_checked_slicing", since = "1.20.0")]
    impl SliceIndex<str> for ops::RangeFrom<usize> {
        type Output = str;
        #[inline]
        fn get(self, slice: &str) -> Option<&Self::Output> {
            if slice.is_char_boundary(self.start) {
                Some(unsafe { self.get_unchecked(slice) })
            } else {
                None
            }
        }
        #[inline]
        fn get_mut(self, slice: &mut str) -> Option<&mut Self::Output> {
            if slice.is_char_boundary(self.start) {
                Some(unsafe { self.get_unchecked_mut(slice) })
            } else {
                None
            }
        }
        #[inline]
        unsafe fn get_unchecked(self, slice: &str) -> &Self::Output {
            let ptr = slice.as_ptr().add(self.start);
            let len = slice.len() - self.start;
            super::from_utf8_unchecked(slice::from_raw_parts(ptr, len))
        }
        #[inline]
        unsafe fn get_unchecked_mut(self, slice: &mut str) -> &mut Self::Output {
            let ptr = slice.as_ptr().add(self.start);
            let len = slice.len() - self.start;
            super::from_utf8_unchecked_mut(slice::from_raw_parts_mut(ptr as *mut u8, len))
        }
        #[inline]
        fn index(self, slice: &str) -> &Self::Output {
            let (start, end) = (self.start, slice.len());
            self.get(slice).unwrap_or_else(|| super::slice_error_fail(slice, start, end))
        }
        #[inline]
        fn index_mut(self, slice: &mut str) -> &mut Self::Output {
            // is_char_boundary checks that the index is in [0, .len()]
            if slice.is_char_boundary(self.start) {
                unsafe { self.get_unchecked_mut(slice) }
            } else {
                super::slice_error_fail(slice, self.start, slice.len())
            }
        }
    }

    /// Implements substring slicing with syntax `&self[begin ..= end]` or `&mut
    /// self[begin ..= end]`.
    ///
    /// Returns a slice of the given string from the byte range
    /// [`begin`, `end`]. Equivalent to `&self [begin .. end + 1]` or `&mut
    /// self[begin .. end + 1]`, except if `end` has the maximum value for
    /// `usize`.
    ///
    /// This operation is `O(1)`.
    ///
    /// # Panics
    ///
    /// Panics if `begin` does not point to the starting byte offset of
    /// a character (as defined by `is_char_boundary`), if `end` does not point
    /// to the ending byte offset of a character (`end + 1` is either a starting
    /// byte offset or equal to `len`), if `begin > end`, or if `end >= len`.
    #[stable(feature = "inclusive_range", since = "1.26.0")]
    impl SliceIndex<str> for ops::RangeInclusive<usize> {
        type Output = str;
        #[inline]
        fn get(self, slice: &str) -> Option<&Self::Output> {
            if *self.end() == usize::max_value() { None }
            else { (*self.start()..self.end()+1).get(slice) }
        }
        #[inline]
        fn get_mut(self, slice: &mut str) -> Option<&mut Self::Output> {
            if *self.end() == usize::max_value() { None }
            else { (*self.start()..self.end()+1).get_mut(slice) }
        }
        #[inline]
        unsafe fn get_unchecked(self, slice: &str) -> &Self::Output {
            (*self.start()..self.end()+1).get_unchecked(slice)
        }
        #[inline]
        unsafe fn get_unchecked_mut(self, slice: &mut str) -> &mut Self::Output {
            (*self.start()..self.end()+1).get_unchecked_mut(slice)
        }
        #[inline]
        fn index(self, slice: &str) -> &Self::Output {
            if *self.end() == usize::max_value() { str_index_overflow_fail(); }
            (*self.start()..self.end()+1).index(slice)
        }
        #[inline]
        fn index_mut(self, slice: &mut str) -> &mut Self::Output {
            if *self.end() == usize::max_value() { str_index_overflow_fail(); }
            (*self.start()..self.end()+1).index_mut(slice)
        }
    }

    /// Implements substring slicing with syntax `&self[..= end]` or `&mut
    /// self[..= end]`.
    ///
    /// Returns a slice of the given string from the byte range [0, `end`].
    /// Equivalent to `&self [0 .. end + 1]`, except if `end` has the maximum
    /// value for `usize`.
    ///
    /// This operation is `O(1)`.
    ///
    /// # Panics
    ///
    /// Panics if `end` does not point to the ending byte offset of a character
    /// (`end + 1` is either a starting byte offset as defined by
    /// `is_char_boundary`, or equal to `len`), or if `end >= len`.
    #[stable(feature = "inclusive_range", since = "1.26.0")]
    impl SliceIndex<str> for ops::RangeToInclusive<usize> {
        type Output = str;
        #[inline]
        fn get(self, slice: &str) -> Option<&Self::Output> {
            if self.end == usize::max_value() { None }
            else { (..self.end+1).get(slice) }
        }
        #[inline]
        fn get_mut(self, slice: &mut str) -> Option<&mut Self::Output> {
            if self.end == usize::max_value() { None }
            else { (..self.end+1).get_mut(slice) }
        }
        #[inline]
        unsafe fn get_unchecked(self, slice: &str) -> &Self::Output {
            (..self.end+1).get_unchecked(slice)
        }
        #[inline]
        unsafe fn get_unchecked_mut(self, slice: &mut str) -> &mut Self::Output {
            (..self.end+1).get_unchecked_mut(slice)
        }
        #[inline]
        fn index(self, slice: &str) -> &Self::Output {
            if self.end == usize::max_value() { str_index_overflow_fail(); }
            (..self.end+1).index(slice)
        }
        #[inline]
        fn index_mut(self, slice: &mut str) -> &mut Self::Output {
            if self.end == usize::max_value() { str_index_overflow_fail(); }
            (..self.end+1).index_mut(slice)
        }
    }
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

#[lang = "str"]
#[cfg(not(test))]
impl str {
    /// Returns the length of `self`.
    ///
    /// This length is in bytes, not [`char`]s or graphemes. In other words,
    /// it may not be what a human considers the length of the string.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let len = "foo".len();
    /// assert_eq!(3, len);
    ///
    /// let len = "ƒoo".len(); // fancy f!
    /// assert_eq!(4, len);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    #[rustc_const_unstable(feature = "const_str_len")]
    pub const fn len(&self) -> usize {
        self.as_bytes().len()
    }

    /// Returns `true` if `self` has a length of zero bytes.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = "";
    /// assert!(s.is_empty());
    ///
    /// let s = "not empty";
    /// assert!(!s.is_empty());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_const_unstable(feature = "const_str_len")]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Checks that `index`-th byte lies at the start and/or end of a
    /// UTF-8 code point sequence.
    ///
    /// The start and end of the string (when `index == self.len()`) are
    /// considered to be
    /// boundaries.
    ///
    /// Returns `false` if `index` is greater than `self.len()`.
    ///
    /// # Examples
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    /// assert!(s.is_char_boundary(0));
    /// // start of `老`
    /// assert!(s.is_char_boundary(6));
    /// assert!(s.is_char_boundary(s.len()));
    ///
    /// // second byte of `ö`
    /// assert!(!s.is_char_boundary(2));
    ///
    /// // third byte of `老`
    /// assert!(!s.is_char_boundary(8));
    /// ```
    #[stable(feature = "is_char_boundary", since = "1.9.0")]
    #[inline]
    pub fn is_char_boundary(&self, index: usize) -> bool {
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

    /// Converts a string slice to a byte slice. To convert the byte slice back
    /// into a string slice, use the [`str::from_utf8`] function.
    ///
    /// [`str::from_utf8`]: ./str/fn.from_utf8.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let bytes = "bors".as_bytes();
    /// assert_eq!(b"bors", bytes);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline(always)]
    #[rustc_const_unstable(feature="const_str_as_bytes")]
    pub const fn as_bytes(&self) -> &[u8] {
        union Slices<'a> {
            str: &'a str,
            slice: &'a [u8],
        }
        unsafe { Slices { str: self }.slice }
    }

    /// Converts a mutable string slice to a mutable byte slice. To convert the
    /// mutable byte slice back into a mutable string slice, use the
    /// [`str::from_utf8_mut`] function.
    ///
    /// [`str::from_utf8_mut`]: ./str/fn.from_utf8_mut.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let mut s = String::from("Hello");
    /// let bytes = unsafe { s.as_bytes_mut() };
    ///
    /// assert_eq!(b"Hello", bytes);
    /// ```
    ///
    /// Mutability:
    ///
    /// ```
    /// let mut s = String::from("🗻∈🌏");
    ///
    /// unsafe {
    ///     let bytes = s.as_bytes_mut();
    ///
    ///     bytes[0] = 0xF0;
    ///     bytes[1] = 0x9F;
    ///     bytes[2] = 0x8D;
    ///     bytes[3] = 0x94;
    /// }
    ///
    /// assert_eq!("🍔∈🌏", s);
    /// ```
    #[stable(feature = "str_mut_extras", since = "1.20.0")]
    #[inline(always)]
    pub unsafe fn as_bytes_mut(&mut self) -> &mut [u8] {
        &mut *(self as *mut str as *mut [u8])
    }

    /// Converts a string slice to a raw pointer.
    ///
    /// As string slices are a slice of bytes, the raw pointer points to a
    /// [`u8`]. This pointer will be pointing to the first byte of the string
    /// slice.
    ///
    /// [`u8`]: primitive.u8.html
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = "Hello";
    /// let ptr = s.as_ptr();
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub const fn as_ptr(&self) -> *const u8 {
        self as *const str as *const u8
    }

    /// Returns a subslice of `str`.
    ///
    /// This is the non-panicking alternative to indexing the `str`. Returns
    /// [`None`] whenever equivalent indexing operation would panic.
    ///
    /// [`None`]: option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// ```
    /// let v = String::from("🗻∈🌏");
    ///
    /// assert_eq!(Some("🗻"), v.get(0..4));
    ///
    /// // indices not on UTF-8 sequence boundaries
    /// assert!(v.get(1..).is_none());
    /// assert!(v.get(..8).is_none());
    ///
    /// // out of bounds
    /// assert!(v.get(..42).is_none());
    /// ```
    #[stable(feature = "str_checked_slicing", since = "1.20.0")]
    #[inline]
    pub fn get<I: SliceIndex<str>>(&self, i: I) -> Option<&I::Output> {
        i.get(self)
    }

    /// Returns a mutable subslice of `str`.
    ///
    /// This is the non-panicking alternative to indexing the `str`. Returns
    /// [`None`] whenever equivalent indexing operation would panic.
    ///
    /// [`None`]: option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = String::from("hello");
    /// // correct length
    /// assert!(v.get_mut(0..5).is_some());
    /// // out of bounds
    /// assert!(v.get_mut(..42).is_none());
    /// assert_eq!(Some("he"), v.get_mut(0..2).map(|v| &*v));
    ///
    /// assert_eq!("hello", v);
    /// {
    ///     let s = v.get_mut(0..2);
    ///     let s = s.map(|s| {
    ///         s.make_ascii_uppercase();
    ///         &*s
    ///     });
    ///     assert_eq!(Some("HE"), s);
    /// }
    /// assert_eq!("HEllo", v);
    /// ```
    #[stable(feature = "str_checked_slicing", since = "1.20.0")]
    #[inline]
    pub fn get_mut<I: SliceIndex<str>>(&mut self, i: I) -> Option<&mut I::Output> {
        i.get_mut(self)
    }

    /// Returns a unchecked subslice of `str`.
    ///
    /// This is the unchecked alternative to indexing the `str`.
    ///
    /// # Safety
    ///
    /// Callers of this function are responsible that these preconditions are
    /// satisfied:
    ///
    /// * The starting index must come before the ending index;
    /// * Indexes must be within bounds of the original slice;
    /// * Indexes must lie on UTF-8 sequence boundaries.
    ///
    /// Failing that, the returned string slice may reference invalid memory or
    /// violate the invariants communicated by the `str` type.
    ///
    /// # Examples
    ///
    /// ```
    /// let v = "🗻∈🌏";
    /// unsafe {
    ///     assert_eq!("🗻", v.get_unchecked(0..4));
    ///     assert_eq!("∈", v.get_unchecked(4..7));
    ///     assert_eq!("🌏", v.get_unchecked(7..11));
    /// }
    /// ```
    #[stable(feature = "str_checked_slicing", since = "1.20.0")]
    #[inline]
    pub unsafe fn get_unchecked<I: SliceIndex<str>>(&self, i: I) -> &I::Output {
        i.get_unchecked(self)
    }

    /// Returns a mutable, unchecked subslice of `str`.
    ///
    /// This is the unchecked alternative to indexing the `str`.
    ///
    /// # Safety
    ///
    /// Callers of this function are responsible that these preconditions are
    /// satisfied:
    ///
    /// * The starting index must come before the ending index;
    /// * Indexes must be within bounds of the original slice;
    /// * Indexes must lie on UTF-8 sequence boundaries.
    ///
    /// Failing that, the returned string slice may reference invalid memory or
    /// violate the invariants communicated by the `str` type.
    ///
    /// # Examples
    ///
    /// ```
    /// let mut v = String::from("🗻∈🌏");
    /// unsafe {
    ///     assert_eq!("🗻", v.get_unchecked_mut(0..4));
    ///     assert_eq!("∈", v.get_unchecked_mut(4..7));
    ///     assert_eq!("🌏", v.get_unchecked_mut(7..11));
    /// }
    /// ```
    #[stable(feature = "str_checked_slicing", since = "1.20.0")]
    #[inline]
    pub unsafe fn get_unchecked_mut<I: SliceIndex<str>>(&mut self, i: I) -> &mut I::Output {
        i.get_unchecked_mut(self)
    }

    /// Creates a string slice from another string slice, bypassing safety
    /// checks.
    ///
    /// This is generally not recommended, use with caution! For a safe
    /// alternative see [`str`] and [`Index`].
    ///
    /// [`str`]: primitive.str.html
    /// [`Index`]: ops/trait.Index.html
    ///
    /// This new slice goes from `begin` to `end`, including `begin` but
    /// excluding `end`.
    ///
    /// To get a mutable string slice instead, see the
    /// [`slice_mut_unchecked`] method.
    ///
    /// [`slice_mut_unchecked`]: #method.slice_mut_unchecked
    ///
    /// # Safety
    ///
    /// Callers of this function are responsible that three preconditions are
    /// satisfied:
    ///
    /// * `begin` must come before `end`.
    /// * `begin` and `end` must be byte positions within the string slice.
    /// * `begin` and `end` must lie on UTF-8 sequence boundaries.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// unsafe {
    ///     assert_eq!("Löwe 老虎 Léopard", s.slice_unchecked(0, 21));
    /// }
    ///
    /// let s = "Hello, world!";
    ///
    /// unsafe {
    ///     assert_eq!("world", s.slice_unchecked(7, 12));
    /// }
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_deprecated(since = "1.29.0", reason = "use `get_unchecked(begin..end)` instead")]
    #[inline]
    pub unsafe fn slice_unchecked(&self, begin: usize, end: usize) -> &str {
        (begin..end).get_unchecked(self)
    }

    /// Creates a string slice from another string slice, bypassing safety
    /// checks.
    /// This is generally not recommended, use with caution! For a safe
    /// alternative see [`str`] and [`IndexMut`].
    ///
    /// [`str`]: primitive.str.html
    /// [`IndexMut`]: ops/trait.IndexMut.html
    ///
    /// This new slice goes from `begin` to `end`, including `begin` but
    /// excluding `end`.
    ///
    /// To get an immutable string slice instead, see the
    /// [`slice_unchecked`] method.
    ///
    /// [`slice_unchecked`]: #method.slice_unchecked
    ///
    /// # Safety
    ///
    /// Callers of this function are responsible that three preconditions are
    /// satisfied:
    ///
    /// * `begin` must come before `end`.
    /// * `begin` and `end` must be byte positions within the string slice.
    /// * `begin` and `end` must lie on UTF-8 sequence boundaries.
    #[stable(feature = "str_slice_mut", since = "1.5.0")]
    #[rustc_deprecated(since = "1.29.0", reason = "use `get_unchecked_mut(begin..end)` instead")]
    #[inline]
    pub unsafe fn slice_mut_unchecked(&mut self, begin: usize, end: usize) -> &mut str {
        (begin..end).get_unchecked_mut(self)
    }

    /// Divide one string slice into two at an index.
    ///
    /// The argument, `mid`, should be a byte offset from the start of the
    /// string. It must also be on the boundary of a UTF-8 code point.
    ///
    /// The two slices returned go from the start of the string slice to `mid`,
    /// and from `mid` to the end of the string slice.
    ///
    /// To get mutable string slices instead, see the [`split_at_mut`]
    /// method.
    ///
    /// [`split_at_mut`]: #method.split_at_mut
    ///
    /// # Panics
    ///
    /// Panics if `mid` is not on a UTF-8 code point boundary, or if it is
    /// beyond the last code point of the string slice.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = "Per Martin-Löf";
    ///
    /// let (first, last) = s.split_at(3);
    ///
    /// assert_eq!("Per", first);
    /// assert_eq!(" Martin-Löf", last);
    /// ```
    #[inline]
    #[stable(feature = "str_split_at", since = "1.4.0")]
    pub fn split_at(&self, mid: usize) -> (&str, &str) {
        // is_char_boundary checks that the index is in [0, .len()]
        if self.is_char_boundary(mid) {
            unsafe {
                (self.get_unchecked(0..mid),
                 self.get_unchecked(mid..self.len()))
            }
        } else {
            slice_error_fail(self, 0, mid)
        }
    }

    /// Divide one mutable string slice into two at an index.
    ///
    /// The argument, `mid`, should be a byte offset from the start of the
    /// string. It must also be on the boundary of a UTF-8 code point.
    ///
    /// The two slices returned go from the start of the string slice to `mid`,
    /// and from `mid` to the end of the string slice.
    ///
    /// To get immutable string slices instead, see the [`split_at`] method.
    ///
    /// [`split_at`]: #method.split_at
    ///
    /// # Panics
    ///
    /// Panics if `mid` is not on a UTF-8 code point boundary, or if it is
    /// beyond the last code point of the string slice.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let mut s = "Per Martin-Löf".to_string();
    /// {
    ///     let (first, last) = s.split_at_mut(3);
    ///     first.make_ascii_uppercase();
    ///     assert_eq!("PER", first);
    ///     assert_eq!(" Martin-Löf", last);
    /// }
    /// assert_eq!("PER Martin-Löf", s);
    /// ```
    #[inline]
    #[stable(feature = "str_split_at", since = "1.4.0")]
    pub fn split_at_mut(&mut self, mid: usize) -> (&mut str, &mut str) {
        // is_char_boundary checks that the index is in [0, .len()]
        if self.is_char_boundary(mid) {
            let len = self.len();
            let ptr = self.as_ptr() as *mut u8;
            unsafe {
                (from_utf8_unchecked_mut(slice::from_raw_parts_mut(ptr, mid)),
                 from_utf8_unchecked_mut(slice::from_raw_parts_mut(
                    ptr.add(mid),
                    len - mid
                 )))
            }
        } else {
            slice_error_fail(self, 0, mid)
        }
    }

    /// Returns an iterator over the [`char`]s of a string slice.
    ///
    /// As a string slice consists of valid UTF-8, we can iterate through a
    /// string slice by [`char`]. This method returns such an iterator.
    ///
    /// It's important to remember that [`char`] represents a Unicode Scalar
    /// Value, and may not match your idea of what a 'character' is. Iteration
    /// over grapheme clusters may be what you actually want.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let word = "goodbye";
    ///
    /// let count = word.chars().count();
    /// assert_eq!(7, count);
    ///
    /// let mut chars = word.chars();
    ///
    /// assert_eq!(Some('g'), chars.next());
    /// assert_eq!(Some('o'), chars.next());
    /// assert_eq!(Some('o'), chars.next());
    /// assert_eq!(Some('d'), chars.next());
    /// assert_eq!(Some('b'), chars.next());
    /// assert_eq!(Some('y'), chars.next());
    /// assert_eq!(Some('e'), chars.next());
    ///
    /// assert_eq!(None, chars.next());
    /// ```
    ///
    /// Remember, [`char`]s may not match your human intuition about characters:
    ///
    /// ```
    /// let y = "y̆";
    ///
    /// let mut chars = y.chars();
    ///
    /// assert_eq!(Some('y'), chars.next()); // not 'y̆'
    /// assert_eq!(Some('\u{0306}'), chars.next());
    ///
    /// assert_eq!(None, chars.next());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn chars(&self) -> Chars {
        Chars{iter: self.as_bytes().iter()}
    }

    /// Returns an iterator over the [`char`]s of a string slice, and their
    /// positions.
    ///
    /// As a string slice consists of valid UTF-8, we can iterate through a
    /// string slice by [`char`]. This method returns an iterator of both
    /// these [`char`]s, as well as their byte positions.
    ///
    /// The iterator yields tuples. The position is first, the [`char`] is
    /// second.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let word = "goodbye";
    ///
    /// let count = word.char_indices().count();
    /// assert_eq!(7, count);
    ///
    /// let mut char_indices = word.char_indices();
    ///
    /// assert_eq!(Some((0, 'g')), char_indices.next());
    /// assert_eq!(Some((1, 'o')), char_indices.next());
    /// assert_eq!(Some((2, 'o')), char_indices.next());
    /// assert_eq!(Some((3, 'd')), char_indices.next());
    /// assert_eq!(Some((4, 'b')), char_indices.next());
    /// assert_eq!(Some((5, 'y')), char_indices.next());
    /// assert_eq!(Some((6, 'e')), char_indices.next());
    ///
    /// assert_eq!(None, char_indices.next());
    /// ```
    ///
    /// Remember, [`char`]s may not match your human intuition about characters:
    ///
    /// ```
    /// let yes = "y̆es";
    ///
    /// let mut char_indices = yes.char_indices();
    ///
    /// assert_eq!(Some((0, 'y')), char_indices.next()); // not (0, 'y̆')
    /// assert_eq!(Some((1, '\u{0306}')), char_indices.next());
    ///
    /// // note the 3 here - the last character took up two bytes
    /// assert_eq!(Some((3, 'e')), char_indices.next());
    /// assert_eq!(Some((4, 's')), char_indices.next());
    ///
    /// assert_eq!(None, char_indices.next());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn char_indices(&self) -> CharIndices {
        CharIndices { front_offset: 0, iter: self.chars() }
    }

    /// An iterator over the bytes of a string slice.
    ///
    /// As a string slice consists of a sequence of bytes, we can iterate
    /// through a string slice by byte. This method returns such an iterator.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let mut bytes = "bors".bytes();
    ///
    /// assert_eq!(Some(b'b'), bytes.next());
    /// assert_eq!(Some(b'o'), bytes.next());
    /// assert_eq!(Some(b'r'), bytes.next());
    /// assert_eq!(Some(b's'), bytes.next());
    ///
    /// assert_eq!(None, bytes.next());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn bytes(&self) -> Bytes {
        Bytes(self.as_bytes().iter().cloned())
    }

    /// Split a string slice by whitespace.
    ///
    /// The iterator returned will return string slices that are sub-slices of
    /// the original string slice, separated by any amount of whitespace.
    ///
    /// 'Whitespace' is defined according to the terms of the Unicode Derived
    /// Core Property `White_Space`. If you only want to split on ASCII whitespace
    /// instead, use [`split_ascii_whitespace`].
    ///
    /// [`split_ascii_whitespace`]: #method.split_ascii_whitespace
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let mut iter = "A few words".split_whitespace();
    ///
    /// assert_eq!(Some("A"), iter.next());
    /// assert_eq!(Some("few"), iter.next());
    /// assert_eq!(Some("words"), iter.next());
    ///
    /// assert_eq!(None, iter.next());
    /// ```
    ///
    /// All kinds of whitespace are considered:
    ///
    /// ```
    /// let mut iter = " Mary   had\ta\u{2009}little  \n\t lamb".split_whitespace();
    /// assert_eq!(Some("Mary"), iter.next());
    /// assert_eq!(Some("had"), iter.next());
    /// assert_eq!(Some("a"), iter.next());
    /// assert_eq!(Some("little"), iter.next());
    /// assert_eq!(Some("lamb"), iter.next());
    ///
    /// assert_eq!(None, iter.next());
    /// ```
    #[stable(feature = "split_whitespace", since = "1.1.0")]
    #[inline]
    pub fn split_whitespace(&self) -> SplitWhitespace {
        SplitWhitespace { inner: self.split(IsWhitespace).filter(IsNotEmpty) }
    }

    /// Split a string slice by ASCII whitespace.
    ///
    /// The iterator returned will return string slices that are sub-slices of
    /// the original string slice, separated by any amount of ASCII whitespace.
    ///
    /// To split by Unicode `Whitespace` instead, use [`split_whitespace`].
    ///
    /// [`split_whitespace`]: #method.split_whitespace
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// #![feature(split_ascii_whitespace)]
    /// let mut iter = "A few words".split_ascii_whitespace();
    ///
    /// assert_eq!(Some("A"), iter.next());
    /// assert_eq!(Some("few"), iter.next());
    /// assert_eq!(Some("words"), iter.next());
    ///
    /// assert_eq!(None, iter.next());
    /// ```
    ///
    /// All kinds of ASCII whitespace are considered:
    ///
    /// ```
    /// let mut iter = " Mary   had\ta little  \n\t lamb".split_whitespace();
    /// assert_eq!(Some("Mary"), iter.next());
    /// assert_eq!(Some("had"), iter.next());
    /// assert_eq!(Some("a"), iter.next());
    /// assert_eq!(Some("little"), iter.next());
    /// assert_eq!(Some("lamb"), iter.next());
    ///
    /// assert_eq!(None, iter.next());
    /// ```
    #[unstable(feature = "split_ascii_whitespace", issue = "48656")]
    #[inline]
    pub fn split_ascii_whitespace(&self) -> SplitAsciiWhitespace {
        let inner = self
            .as_bytes()
            .split(IsAsciiWhitespace)
            .filter(IsNotEmpty)
            .map(UnsafeBytesToStr);
        SplitAsciiWhitespace { inner }
    }

    /// An iterator over the lines of a string, as string slices.
    ///
    /// Lines are ended with either a newline (`\n`) or a carriage return with
    /// a line feed (`\r\n`).
    ///
    /// The final line ending is optional.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let text = "foo\r\nbar\n\nbaz\n";
    /// let mut lines = text.lines();
    ///
    /// assert_eq!(Some("foo"), lines.next());
    /// assert_eq!(Some("bar"), lines.next());
    /// assert_eq!(Some(""), lines.next());
    /// assert_eq!(Some("baz"), lines.next());
    ///
    /// assert_eq!(None, lines.next());
    /// ```
    ///
    /// The final line ending isn't required:
    ///
    /// ```
    /// let text = "foo\nbar\n\r\nbaz";
    /// let mut lines = text.lines();
    ///
    /// assert_eq!(Some("foo"), lines.next());
    /// assert_eq!(Some("bar"), lines.next());
    /// assert_eq!(Some(""), lines.next());
    /// assert_eq!(Some("baz"), lines.next());
    ///
    /// assert_eq!(None, lines.next());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn lines(&self) -> Lines {
        Lines(self.split_terminator('\n').map(LinesAnyMap))
    }

    /// An iterator over the lines of a string.
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_deprecated(since = "1.4.0", reason = "use lines() instead now")]
    #[inline]
    #[allow(deprecated)]
    pub fn lines_any(&self) -> LinesAny {
        LinesAny(self.lines())
    }

    /// Returns an iterator of `u16` over the string encoded as UTF-16.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let text = "Zażółć gęślą jaźń";
    ///
    /// let utf8_len = text.len();
    /// let utf16_len = text.encode_utf16().count();
    ///
    /// assert!(utf16_len <= utf8_len);
    /// ```
    #[stable(feature = "encode_utf16", since = "1.8.0")]
    pub fn encode_utf16(&self) -> EncodeUtf16 {
        EncodeUtf16 { chars: self.chars(), extra: 0 }
    }

    /// Returns `true` if the given pattern matches a sub-slice of
    /// this string slice.
    ///
    /// Returns `false` if it does not.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let bananas = "bananas";
    ///
    /// assert!(bananas.contains("nana"));
    /// assert!(!bananas.contains("apples"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn contains<'a, P: Pattern<'a>>(&'a self, pat: P) -> bool {
        pat.is_contained_in(self)
    }

    /// Returns `true` if the given pattern matches a prefix of this
    /// string slice.
    ///
    /// Returns `false` if it does not.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let bananas = "bananas";
    ///
    /// assert!(bananas.starts_with("bana"));
    /// assert!(!bananas.starts_with("nana"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn starts_with<'a, P: Pattern<'a>>(&'a self, pat: P) -> bool {
        pat.is_prefix_of(self)
    }

    /// Returns `true` if the given pattern matches a suffix of this
    /// string slice.
    ///
    /// Returns `false` if it does not.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let bananas = "bananas";
    ///
    /// assert!(bananas.ends_with("anas"));
    /// assert!(!bananas.ends_with("nana"));
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn ends_with<'a, P: Pattern<'a>>(&'a self, pat: P) -> bool
        where P::Searcher: ReverseSearcher<'a>
    {
        pat.is_suffix_of(self)
    }

    /// Returns the byte index of the first character of this string slice that
    /// matches the pattern.
    ///
    /// Returns [`None`] if the pattern doesn't match.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines if
    /// a character matches.
    ///
    /// [`None`]: option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// assert_eq!(s.find('L'), Some(0));
    /// assert_eq!(s.find('é'), Some(14));
    /// assert_eq!(s.find("Léopard"), Some(13));
    /// ```
    ///
    /// More complex patterns using point-free style and closures:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// assert_eq!(s.find(char::is_whitespace), Some(5));
    /// assert_eq!(s.find(char::is_lowercase), Some(1));
    /// assert_eq!(s.find(|c: char| c.is_whitespace() || c.is_lowercase()), Some(1));
    /// assert_eq!(s.find(|c: char| (c < 'o') && (c > 'a')), Some(4));
    /// ```
    ///
    /// Not finding the pattern:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    /// let x: &[_] = &['1', '2'];
    ///
    /// assert_eq!(s.find(x), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn find<'a, P: Pattern<'a>>(&'a self, pat: P) -> Option<usize> {
        pat.into_searcher(self).next_match().map(|(i, _)| i)
    }

    /// Returns the byte index of the last character of this string slice that
    /// matches the pattern.
    ///
    /// Returns [`None`] if the pattern doesn't match.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines if
    /// a character matches.
    ///
    /// [`None`]: option/enum.Option.html#variant.None
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// assert_eq!(s.rfind('L'), Some(13));
    /// assert_eq!(s.rfind('é'), Some(14));
    /// ```
    ///
    /// More complex patterns with closures:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    ///
    /// assert_eq!(s.rfind(char::is_whitespace), Some(12));
    /// assert_eq!(s.rfind(char::is_lowercase), Some(20));
    /// ```
    ///
    /// Not finding the pattern:
    ///
    /// ```
    /// let s = "Löwe 老虎 Léopard";
    /// let x: &[_] = &['1', '2'];
    ///
    /// assert_eq!(s.rfind(x), None);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn rfind<'a, P: Pattern<'a>>(&'a self, pat: P) -> Option<usize>
        where P::Searcher: ReverseSearcher<'a>
    {
        pat.into_searcher(self).next_match_back().map(|(i, _)| i)
    }

    /// An iterator over substrings of this string slice, separated by
    /// characters matched by a pattern.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines the
    /// split.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will be a [`DoubleEndedIterator`] if the pattern
    /// allows a reverse search and forward/reverse search yields the same
    /// elements. This is true for, eg, [`char`] but not for `&str`.
    ///
    /// [`DoubleEndedIterator`]: iter/trait.DoubleEndedIterator.html
    ///
    /// If the pattern allows a reverse search but its results might differ
    /// from a forward search, the [`rsplit`] method can be used.
    ///
    /// [`rsplit`]: #method.rsplit
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// let v: Vec<&str> = "Mary had a little lamb".split(' ').collect();
    /// assert_eq!(v, ["Mary", "had", "a", "little", "lamb"]);
    ///
    /// let v: Vec<&str> = "".split('X').collect();
    /// assert_eq!(v, [""]);
    ///
    /// let v: Vec<&str> = "lionXXtigerXleopard".split('X').collect();
    /// assert_eq!(v, ["lion", "", "tiger", "leopard"]);
    ///
    /// let v: Vec<&str> = "lion::tiger::leopard".split("::").collect();
    /// assert_eq!(v, ["lion", "tiger", "leopard"]);
    ///
    /// let v: Vec<&str> = "abc1def2ghi".split(char::is_numeric).collect();
    /// assert_eq!(v, ["abc", "def", "ghi"]);
    ///
    /// let v: Vec<&str> = "lionXtigerXleopard".split(char::is_uppercase).collect();
    /// assert_eq!(v, ["lion", "tiger", "leopard"]);
    /// ```
    ///
    /// A more complex pattern, using a closure:
    ///
    /// ```
    /// let v: Vec<&str> = "abc1defXghi".split(|c| c == '1' || c == 'X').collect();
    /// assert_eq!(v, ["abc", "def", "ghi"]);
    /// ```
    ///
    /// If a string contains multiple contiguous separators, you will end up
    /// with empty strings in the output:
    ///
    /// ```
    /// let x = "||||a||b|c".to_string();
    /// let d: Vec<_> = x.split('|').collect();
    ///
    /// assert_eq!(d, &["", "", "", "", "a", "", "b", "c"]);
    /// ```
    ///
    /// Contiguous separators are separated by the empty string.
    ///
    /// ```
    /// let x = "(///)".to_string();
    /// let d: Vec<_> = x.split('/').collect();
    ///
    /// assert_eq!(d, &["(", "", "", ")"]);
    /// ```
    ///
    /// Separators at the start or end of a string are neighbored
    /// by empty strings.
    ///
    /// ```
    /// let d: Vec<_> = "010".split("0").collect();
    /// assert_eq!(d, &["", "1", ""]);
    /// ```
    ///
    /// When the empty string is used as a separator, it separates
    /// every character in the string, along with the beginning
    /// and end of the string.
    ///
    /// ```
    /// let f: Vec<_> = "rust".split("").collect();
    /// assert_eq!(f, &["", "r", "u", "s", "t", ""]);
    /// ```
    ///
    /// Contiguous separators can lead to possibly surprising behavior
    /// when whitespace is used as the separator. This code is correct:
    ///
    /// ```
    /// let x = "    a  b c".to_string();
    /// let d: Vec<_> = x.split(' ').collect();
    ///
    /// assert_eq!(d, &["", "", "", "", "a", "", "b", "c"]);
    /// ```
    ///
    /// It does _not_ give you:
    ///
    /// ```,ignore
    /// assert_eq!(d, &["a", "b", "c"]);
    /// ```
    ///
    /// Use [`split_whitespace`] for this behavior.
    ///
    /// [`split_whitespace`]: #method.split_whitespace
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn split<'a, P: Pattern<'a>>(&'a self, pat: P) -> Split<'a, P> {
        Split(SplitInternal {
            start: 0,
            end: self.len(),
            matcher: pat.into_searcher(self),
            allow_trailing_empty: true,
            finished: false,
        })
    }

    /// An iterator over substrings of the given string slice, separated by
    /// characters matched by a pattern and yielded in reverse order.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines the
    /// split.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator requires that the pattern supports a reverse
    /// search, and it will be a [`DoubleEndedIterator`] if a forward/reverse
    /// search yields the same elements.
    ///
    /// [`DoubleEndedIterator`]: iter/trait.DoubleEndedIterator.html
    ///
    /// For iterating from the front, the [`split`] method can be used.
    ///
    /// [`split`]: #method.split
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// let v: Vec<&str> = "Mary had a little lamb".rsplit(' ').collect();
    /// assert_eq!(v, ["lamb", "little", "a", "had", "Mary"]);
    ///
    /// let v: Vec<&str> = "".rsplit('X').collect();
    /// assert_eq!(v, [""]);
    ///
    /// let v: Vec<&str> = "lionXXtigerXleopard".rsplit('X').collect();
    /// assert_eq!(v, ["leopard", "tiger", "", "lion"]);
    ///
    /// let v: Vec<&str> = "lion::tiger::leopard".rsplit("::").collect();
    /// assert_eq!(v, ["leopard", "tiger", "lion"]);
    /// ```
    ///
    /// A more complex pattern, using a closure:
    ///
    /// ```
    /// let v: Vec<&str> = "abc1defXghi".rsplit(|c| c == '1' || c == 'X').collect();
    /// assert_eq!(v, ["ghi", "def", "abc"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn rsplit<'a, P: Pattern<'a>>(&'a self, pat: P) -> RSplit<'a, P>
        where P::Searcher: ReverseSearcher<'a>
    {
        RSplit(self.split(pat).0)
    }

    /// An iterator over substrings of the given string slice, separated by
    /// characters matched by a pattern.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines the
    /// split.
    ///
    /// Equivalent to [`split`], except that the trailing substring
    /// is skipped if empty.
    ///
    /// [`split`]: #method.split
    ///
    /// This method can be used for string data that is _terminated_,
    /// rather than _separated_ by a pattern.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will be a [`DoubleEndedIterator`] if the pattern
    /// allows a reverse search and forward/reverse search yields the same
    /// elements. This is true for, eg, [`char`] but not for `&str`.
    ///
    /// [`DoubleEndedIterator`]: iter/trait.DoubleEndedIterator.html
    ///
    /// If the pattern allows a reverse search but its results might differ
    /// from a forward search, the [`rsplit_terminator`] method can be used.
    ///
    /// [`rsplit_terminator`]: #method.rsplit_terminator
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let v: Vec<&str> = "A.B.".split_terminator('.').collect();
    /// assert_eq!(v, ["A", "B"]);
    ///
    /// let v: Vec<&str> = "A..B..".split_terminator(".").collect();
    /// assert_eq!(v, ["A", "", "B", ""]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn split_terminator<'a, P: Pattern<'a>>(&'a self, pat: P) -> SplitTerminator<'a, P> {
        SplitTerminator(SplitInternal {
            allow_trailing_empty: false,
            ..self.split(pat).0
        })
    }

    /// An iterator over substrings of `self`, separated by characters
    /// matched by a pattern and yielded in reverse order.
    ///
    /// The pattern can be a simple `&str`, [`char`], or a closure that
    /// determines the split.
    /// Additional libraries might provide more complex patterns like
    /// regular expressions.
    ///
    /// Equivalent to [`split`], except that the trailing substring is
    /// skipped if empty.
    ///
    /// [`split`]: #method.split
    ///
    /// This method can be used for string data that is _terminated_,
    /// rather than _separated_ by a pattern.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator requires that the pattern supports a
    /// reverse search, and it will be double ended if a forward/reverse
    /// search yields the same elements.
    ///
    /// For iterating from the front, the [`split_terminator`] method can be
    /// used.
    ///
    /// [`split_terminator`]: #method.split_terminator
    ///
    /// # Examples
    ///
    /// ```
    /// let v: Vec<&str> = "A.B.".rsplit_terminator('.').collect();
    /// assert_eq!(v, ["B", "A"]);
    ///
    /// let v: Vec<&str> = "A..B..".rsplit_terminator(".").collect();
    /// assert_eq!(v, ["", "B", "", "A"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn rsplit_terminator<'a, P: Pattern<'a>>(&'a self, pat: P) -> RSplitTerminator<'a, P>
        where P::Searcher: ReverseSearcher<'a>
    {
        RSplitTerminator(self.split_terminator(pat).0)
    }

    /// An iterator over substrings of the given string slice, separated by a
    /// pattern, restricted to returning at most `n` items.
    ///
    /// If `n` substrings are returned, the last substring (the `n`th substring)
    /// will contain the remainder of the string.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines the
    /// split.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will not be double ended, because it is
    /// not efficient to support.
    ///
    /// If the pattern allows a reverse search, the [`rsplitn`] method can be
    /// used.
    ///
    /// [`rsplitn`]: #method.rsplitn
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// let v: Vec<&str> = "Mary had a little lambda".splitn(3, ' ').collect();
    /// assert_eq!(v, ["Mary", "had", "a little lambda"]);
    ///
    /// let v: Vec<&str> = "lionXXtigerXleopard".splitn(3, "X").collect();
    /// assert_eq!(v, ["lion", "", "tigerXleopard"]);
    ///
    /// let v: Vec<&str> = "abcXdef".splitn(1, 'X').collect();
    /// assert_eq!(v, ["abcXdef"]);
    ///
    /// let v: Vec<&str> = "".splitn(1, 'X').collect();
    /// assert_eq!(v, [""]);
    /// ```
    ///
    /// A more complex pattern, using a closure:
    ///
    /// ```
    /// let v: Vec<&str> = "abc1defXghi".splitn(2, |c| c == '1' || c == 'X').collect();
    /// assert_eq!(v, ["abc", "defXghi"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn splitn<'a, P: Pattern<'a>>(&'a self, n: usize, pat: P) -> SplitN<'a, P> {
        SplitN(SplitNInternal {
            iter: self.split(pat).0,
            count: n,
        })
    }

    /// An iterator over substrings of this string slice, separated by a
    /// pattern, starting from the end of the string, restricted to returning
    /// at most `n` items.
    ///
    /// If `n` substrings are returned, the last substring (the `n`th substring)
    /// will contain the remainder of the string.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that
    /// determines the split.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will not be double ended, because it is not
    /// efficient to support.
    ///
    /// For splitting from the front, the [`splitn`] method can be used.
    ///
    /// [`splitn`]: #method.splitn
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// let v: Vec<&str> = "Mary had a little lamb".rsplitn(3, ' ').collect();
    /// assert_eq!(v, ["lamb", "little", "Mary had a"]);
    ///
    /// let v: Vec<&str> = "lionXXtigerXleopard".rsplitn(3, 'X').collect();
    /// assert_eq!(v, ["leopard", "tiger", "lionX"]);
    ///
    /// let v: Vec<&str> = "lion::tiger::leopard".rsplitn(2, "::").collect();
    /// assert_eq!(v, ["leopard", "lion::tiger"]);
    /// ```
    ///
    /// A more complex pattern, using a closure:
    ///
    /// ```
    /// let v: Vec<&str> = "abc1defXghi".rsplitn(2, |c| c == '1' || c == 'X').collect();
    /// assert_eq!(v, ["ghi", "abc1def"]);
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[inline]
    pub fn rsplitn<'a, P: Pattern<'a>>(&'a self, n: usize, pat: P) -> RSplitN<'a, P>
        where P::Searcher: ReverseSearcher<'a>
    {
        RSplitN(self.splitn(n, pat).0)
    }

    /// An iterator over the disjoint matches of a pattern within the given string
    /// slice.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that
    /// determines if a character matches.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will be a [`DoubleEndedIterator`] if the pattern
    /// allows a reverse search and forward/reverse search yields the same
    /// elements. This is true for, eg, [`char`] but not for `&str`.
    ///
    /// [`DoubleEndedIterator`]: iter/trait.DoubleEndedIterator.html
    ///
    /// If the pattern allows a reverse search but its results might differ
    /// from a forward search, the [`rmatches`] method can be used.
    ///
    /// [`rmatches`]: #method.rmatches
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let v: Vec<&str> = "abcXXXabcYYYabc".matches("abc").collect();
    /// assert_eq!(v, ["abc", "abc", "abc"]);
    ///
    /// let v: Vec<&str> = "1abc2abc3".matches(char::is_numeric).collect();
    /// assert_eq!(v, ["1", "2", "3"]);
    /// ```
    #[stable(feature = "str_matches", since = "1.2.0")]
    #[inline]
    pub fn matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> Matches<'a, P> {
        Matches(MatchesInternal(pat.into_searcher(self)))
    }

    /// An iterator over the disjoint matches of a pattern within this string slice,
    /// yielded in reverse order.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines if
    /// a character matches.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator requires that the pattern supports a reverse
    /// search, and it will be a [`DoubleEndedIterator`] if a forward/reverse
    /// search yields the same elements.
    ///
    /// [`DoubleEndedIterator`]: iter/trait.DoubleEndedIterator.html
    ///
    /// For iterating from the front, the [`matches`] method can be used.
    ///
    /// [`matches`]: #method.matches
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let v: Vec<&str> = "abcXXXabcYYYabc".rmatches("abc").collect();
    /// assert_eq!(v, ["abc", "abc", "abc"]);
    ///
    /// let v: Vec<&str> = "1abc2abc3".rmatches(char::is_numeric).collect();
    /// assert_eq!(v, ["3", "2", "1"]);
    /// ```
    #[stable(feature = "str_matches", since = "1.2.0")]
    #[inline]
    pub fn rmatches<'a, P: Pattern<'a>>(&'a self, pat: P) -> RMatches<'a, P>
        where P::Searcher: ReverseSearcher<'a>
    {
        RMatches(self.matches(pat).0)
    }

    /// An iterator over the disjoint matches of a pattern within this string
    /// slice as well as the index that the match starts at.
    ///
    /// For matches of `pat` within `self` that overlap, only the indices
    /// corresponding to the first match are returned.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines
    /// if a character matches.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator will be a [`DoubleEndedIterator`] if the pattern
    /// allows a reverse search and forward/reverse search yields the same
    /// elements. This is true for, eg, [`char`] but not for `&str`.
    ///
    /// [`DoubleEndedIterator`]: iter/trait.DoubleEndedIterator.html
    ///
    /// If the pattern allows a reverse search but its results might differ
    /// from a forward search, the [`rmatch_indices`] method can be used.
    ///
    /// [`rmatch_indices`]: #method.rmatch_indices
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let v: Vec<_> = "abcXXXabcYYYabc".match_indices("abc").collect();
    /// assert_eq!(v, [(0, "abc"), (6, "abc"), (12, "abc")]);
    ///
    /// let v: Vec<_> = "1abcabc2".match_indices("abc").collect();
    /// assert_eq!(v, [(1, "abc"), (4, "abc")]);
    ///
    /// let v: Vec<_> = "ababa".match_indices("aba").collect();
    /// assert_eq!(v, [(0, "aba")]); // only the first `aba`
    /// ```
    #[stable(feature = "str_match_indices", since = "1.5.0")]
    #[inline]
    pub fn match_indices<'a, P: Pattern<'a>>(&'a self, pat: P) -> MatchIndices<'a, P> {
        MatchIndices(MatchIndicesInternal(pat.into_searcher(self)))
    }

    /// An iterator over the disjoint matches of a pattern within `self`,
    /// yielded in reverse order along with the index of the match.
    ///
    /// For matches of `pat` within `self` that overlap, only the indices
    /// corresponding to the last match are returned.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines if a
    /// character matches.
    ///
    /// # Iterator behavior
    ///
    /// The returned iterator requires that the pattern supports a reverse
    /// search, and it will be a [`DoubleEndedIterator`] if a forward/reverse
    /// search yields the same elements.
    ///
    /// [`DoubleEndedIterator`]: iter/trait.DoubleEndedIterator.html
    ///
    /// For iterating from the front, the [`match_indices`] method can be used.
    ///
    /// [`match_indices`]: #method.match_indices
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let v: Vec<_> = "abcXXXabcYYYabc".rmatch_indices("abc").collect();
    /// assert_eq!(v, [(12, "abc"), (6, "abc"), (0, "abc")]);
    ///
    /// let v: Vec<_> = "1abcabc2".rmatch_indices("abc").collect();
    /// assert_eq!(v, [(4, "abc"), (1, "abc")]);
    ///
    /// let v: Vec<_> = "ababa".rmatch_indices("aba").collect();
    /// assert_eq!(v, [(2, "aba")]); // only the last `aba`
    /// ```
    #[stable(feature = "str_match_indices", since = "1.5.0")]
    #[inline]
    pub fn rmatch_indices<'a, P: Pattern<'a>>(&'a self, pat: P) -> RMatchIndices<'a, P>
        where P::Searcher: ReverseSearcher<'a>
    {
        RMatchIndices(self.match_indices(pat).0)
    }

    /// Returns a string slice with leading and trailing whitespace removed.
    ///
    /// 'Whitespace' is defined according to the terms of the Unicode Derived
    /// Core Property `White_Space`.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = " Hello\tworld\t";
    ///
    /// assert_eq!("Hello\tworld", s.trim());
    /// ```
    #[must_use = "this returns the trimmed string as a slice, \
                  without modifying the original"]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn trim(&self) -> &str {
        self.trim_matches(|c: char| c.is_whitespace())
    }

    /// Returns a string slice with leading whitespace removed.
    ///
    /// 'Whitespace' is defined according to the terms of the Unicode Derived
    /// Core Property `White_Space`.
    ///
    /// # Text directionality
    ///
    /// A string is a sequence of bytes. `start` in this context means the first
    /// position of that byte string; for a left-to-right language like English or
    /// Russian, this will be left side; and for right-to-left languages like
    /// like Arabic or Hebrew, this will be the right side.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = " Hello\tworld\t";
    /// assert_eq!("Hello\tworld\t", s.trim_start());
    /// ```
    ///
    /// Directionality:
    ///
    /// ```
    /// let s = "  English  ";
    /// assert!(Some('E') == s.trim_start().chars().next());
    ///
    /// let s = "  עברית  ";
    /// assert!(Some('ע') == s.trim_start().chars().next());
    /// ```
    #[must_use = "this returns the trimmed string as a new slice, \
                  without modifying the original"]
    #[stable(feature = "trim_direction", since = "1.30.0")]
    pub fn trim_start(&self) -> &str {
        self.trim_start_matches(|c: char| c.is_whitespace())
    }

    /// Returns a string slice with trailing whitespace removed.
    ///
    /// 'Whitespace' is defined according to the terms of the Unicode Derived
    /// Core Property `White_Space`.
    ///
    /// # Text directionality
    ///
    /// A string is a sequence of bytes. `end` in this context means the last
    /// position of that byte string; for a left-to-right language like English or
    /// Russian, this will be right side; and for right-to-left languages like
    /// like Arabic or Hebrew, this will be the left side.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = " Hello\tworld\t";
    /// assert_eq!(" Hello\tworld", s.trim_end());
    /// ```
    ///
    /// Directionality:
    ///
    /// ```
    /// let s = "  English  ";
    /// assert!(Some('h') == s.trim_end().chars().rev().next());
    ///
    /// let s = "  עברית  ";
    /// assert!(Some('ת') == s.trim_end().chars().rev().next());
    /// ```
    #[must_use = "this returns the trimmed string as a new slice, \
                  without modifying the original"]
    #[stable(feature = "trim_direction", since = "1.30.0")]
    pub fn trim_end(&self) -> &str {
        self.trim_end_matches(|c: char| c.is_whitespace())
    }

    /// Returns a string slice with leading whitespace removed.
    ///
    /// 'Whitespace' is defined according to the terms of the Unicode Derived
    /// Core Property `White_Space`.
    ///
    /// # Text directionality
    ///
    /// A string is a sequence of bytes. 'Left' in this context means the first
    /// position of that byte string; for a language like Arabic or Hebrew
    /// which are 'right to left' rather than 'left to right', this will be
    /// the _right_ side, not the left.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = " Hello\tworld\t";
    ///
    /// assert_eq!("Hello\tworld\t", s.trim_left());
    /// ```
    ///
    /// Directionality:
    ///
    /// ```
    /// let s = "  English";
    /// assert!(Some('E') == s.trim_left().chars().next());
    ///
    /// let s = "  עברית";
    /// assert!(Some('ע') == s.trim_left().chars().next());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_deprecated(reason = "superseded by `trim_start`", since = "1.33.0")]
    pub fn trim_left(&self) -> &str {
        self.trim_start()
    }

    /// Returns a string slice with trailing whitespace removed.
    ///
    /// 'Whitespace' is defined according to the terms of the Unicode Derived
    /// Core Property `White_Space`.
    ///
    /// # Text directionality
    ///
    /// A string is a sequence of bytes. 'Right' in this context means the last
    /// position of that byte string; for a language like Arabic or Hebrew
    /// which are 'right to left' rather than 'left to right', this will be
    /// the _left_ side, not the right.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// let s = " Hello\tworld\t";
    ///
    /// assert_eq!(" Hello\tworld", s.trim_right());
    /// ```
    ///
    /// Directionality:
    ///
    /// ```
    /// let s = "English  ";
    /// assert!(Some('h') == s.trim_right().chars().rev().next());
    ///
    /// let s = "עברית  ";
    /// assert!(Some('ת') == s.trim_right().chars().rev().next());
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_deprecated(reason = "superseded by `trim_end`", since = "1.33.0")]
    pub fn trim_right(&self) -> &str {
        self.trim_end()
    }

    /// Returns a string slice with all prefixes and suffixes that match a
    /// pattern repeatedly removed.
    ///
    /// The pattern can be a [`char`] or a closure that determines if a
    /// character matches.
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// assert_eq!("11foo1bar11".trim_matches('1'), "foo1bar");
    /// assert_eq!("123foo1bar123".trim_matches(char::is_numeric), "foo1bar");
    ///
    /// let x: &[_] = &['1', '2'];
    /// assert_eq!("12foo1bar12".trim_matches(x), "foo1bar");
    /// ```
    ///
    /// A more complex pattern, using a closure:
    ///
    /// ```
    /// assert_eq!("1foo1barXX".trim_matches(|c| c == '1' || c == 'X'), "foo1bar");
    /// ```
    #[must_use = "this returns the trimmed string as a new slice, \
                  without modifying the original"]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn trim_matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> &'a str
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
            self.get_unchecked(i..j)
        }
    }

    /// Returns a string slice with all prefixes that match a pattern
    /// repeatedly removed.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines if
    /// a character matches.
    ///
    /// # Text directionality
    ///
    /// A string is a sequence of bytes. 'Left' in this context means the first
    /// position of that byte string; for a language like Arabic or Hebrew
    /// which are 'right to left' rather than 'left to right', this will be
    /// the _right_ side, not the left.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// assert_eq!("11foo1bar11".trim_start_matches('1'), "foo1bar11");
    /// assert_eq!("123foo1bar123".trim_start_matches(char::is_numeric), "foo1bar123");
    ///
    /// let x: &[_] = &['1', '2'];
    /// assert_eq!("12foo1bar12".trim_start_matches(x), "foo1bar12");
    /// ```
    #[must_use = "this returns the trimmed string as a new slice, \
                  without modifying the original"]
    #[stable(feature = "trim_direction", since = "1.30.0")]
    pub fn trim_start_matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> &'a str {
        let mut i = self.len();
        let mut matcher = pat.into_searcher(self);
        if let Some((a, _)) = matcher.next_reject() {
            i = a;
        }
        unsafe {
            // Searcher is known to return valid indices
            self.get_unchecked(i..self.len())
        }
    }

    /// Returns a string slice with all suffixes that match a pattern
    /// repeatedly removed.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that
    /// determines if a character matches.
    ///
    /// # Text directionality
    ///
    /// A string is a sequence of bytes. 'Right' in this context means the last
    /// position of that byte string; for a language like Arabic or Hebrew
    /// which are 'right to left' rather than 'left to right', this will be
    /// the _left_ side, not the right.
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// assert_eq!("11foo1bar11".trim_end_matches('1'), "11foo1bar");
    /// assert_eq!("123foo1bar123".trim_end_matches(char::is_numeric), "123foo1bar");
    ///
    /// let x: &[_] = &['1', '2'];
    /// assert_eq!("12foo1bar12".trim_end_matches(x), "12foo1bar");
    /// ```
    ///
    /// A more complex pattern, using a closure:
    ///
    /// ```
    /// assert_eq!("1fooX".trim_end_matches(|c| c == '1' || c == 'X'), "1foo");
    /// ```
    #[must_use = "this returns the trimmed string as a new slice, \
                  without modifying the original"]
    #[stable(feature = "trim_direction", since = "1.30.0")]
    pub fn trim_end_matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> &'a str
        where P::Searcher: ReverseSearcher<'a>
    {
        let mut j = 0;
        let mut matcher = pat.into_searcher(self);
        if let Some((_, b)) = matcher.next_reject_back() {
            j = b;
        }
        unsafe {
            // Searcher is known to return valid indices
            self.get_unchecked(0..j)
        }
    }

    /// Returns a string slice with all prefixes that match a pattern
    /// repeatedly removed.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that determines if
    /// a character matches.
    ///
    /// [`char`]: primitive.char.html
    ///
    /// # Text directionality
    ///
    /// A string is a sequence of bytes. 'Left' in this context means the first
    /// position of that byte string; for a language like Arabic or Hebrew
    /// which are 'right to left' rather than 'left to right', this will be
    /// the _right_ side, not the left.
    ///
    /// # Examples
    ///
    /// Basic usage:
    ///
    /// ```
    /// assert_eq!("11foo1bar11".trim_left_matches('1'), "foo1bar11");
    /// assert_eq!("123foo1bar123".trim_left_matches(char::is_numeric), "foo1bar123");
    ///
    /// let x: &[_] = &['1', '2'];
    /// assert_eq!("12foo1bar12".trim_left_matches(x), "foo1bar12");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_deprecated(reason = "superseded by `trim_start_matches`", since = "1.33.0")]
    pub fn trim_left_matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> &'a str {
        self.trim_start_matches(pat)
    }

    /// Returns a string slice with all suffixes that match a pattern
    /// repeatedly removed.
    ///
    /// The pattern can be a `&str`, [`char`], or a closure that
    /// determines if a character matches.
    ///
    /// [`char`]: primitive.char.html
    ///
    /// # Text directionality
    ///
    /// A string is a sequence of bytes. 'Right' in this context means the last
    /// position of that byte string; for a language like Arabic or Hebrew
    /// which are 'right to left' rather than 'left to right', this will be
    /// the _left_ side, not the right.
    ///
    /// # Examples
    ///
    /// Simple patterns:
    ///
    /// ```
    /// assert_eq!("11foo1bar11".trim_right_matches('1'), "11foo1bar");
    /// assert_eq!("123foo1bar123".trim_right_matches(char::is_numeric), "123foo1bar");
    ///
    /// let x: &[_] = &['1', '2'];
    /// assert_eq!("12foo1bar12".trim_right_matches(x), "12foo1bar");
    /// ```
    ///
    /// A more complex pattern, using a closure:
    ///
    /// ```
    /// assert_eq!("1fooX".trim_right_matches(|c| c == '1' || c == 'X'), "1foo");
    /// ```
    #[stable(feature = "rust1", since = "1.0.0")]
    #[rustc_deprecated(reason = "superseded by `trim_end_matches`", since = "1.33.0")]
    pub fn trim_right_matches<'a, P: Pattern<'a>>(&'a self, pat: P) -> &'a str
        where P::Searcher: ReverseSearcher<'a>
    {
        self.trim_end_matches(pat)
    }

    /// Parses this string slice into another type.
    ///
    /// Because `parse` is so general, it can cause problems with type
    /// inference. As such, `parse` is one of the few times you'll see
    /// the syntax affectionately known as the 'turbofish': `::<>`. This
    /// helps the inference algorithm understand specifically which type
    /// you're trying to parse into.
    ///
    /// `parse` can parse any type that implements the [`FromStr`] trait.
    ///
    /// [`FromStr`]: str/trait.FromStr.html
    ///
    /// # Errors
    ///
    /// Will return [`Err`] if it's not possible to parse this string slice into
    /// the desired type.
    ///
    /// [`Err`]: str/trait.FromStr.html#associatedtype.Err
    ///
    /// # Examples
    ///
    /// Basic usage
    ///
    /// ```
    /// let four: u32 = "4".parse().unwrap();
    ///
    /// assert_eq!(4, four);
    /// ```
    ///
    /// Using the 'turbofish' instead of annotating `four`:
    ///
    /// ```
    /// let four = "4".parse::<u32>();
    ///
    /// assert_eq!(Ok(4), four);
    /// ```
    ///
    /// Failing to parse:
    ///
    /// ```
    /// let nope = "j".parse::<u32>();
    ///
    /// assert!(nope.is_err());
    /// ```
    #[inline]
    #[stable(feature = "rust1", since = "1.0.0")]
    pub fn parse<F: FromStr>(&self) -> Result<F, F::Err> {
        FromStr::from_str(self)
    }

    /// Checks if all characters in this string are within the ASCII range.
    ///
    /// # Examples
    ///
    /// ```
    /// let ascii = "hello!\n";
    /// let non_ascii = "Grüße, Jürgen ❤";
    ///
    /// assert!(ascii.is_ascii());
    /// assert!(!non_ascii.is_ascii());
    /// ```
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[inline]
    pub fn is_ascii(&self) -> bool {
        // We can treat each byte as character here: all multibyte characters
        // start with a byte that is not in the ascii range, so we will stop
        // there already.
        self.bytes().all(|b| b.is_ascii())
    }

    /// Checks that two strings are an ASCII case-insensitive match.
    ///
    /// Same as `to_ascii_lowercase(a) == to_ascii_lowercase(b)`,
    /// but without allocating and copying temporaries.
    ///
    /// # Examples
    ///
    /// ```
    /// assert!("Ferris".eq_ignore_ascii_case("FERRIS"));
    /// assert!("Ferrös".eq_ignore_ascii_case("FERRöS"));
    /// assert!(!"Ferrös".eq_ignore_ascii_case("FERRÖS"));
    /// ```
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    #[inline]
    pub fn eq_ignore_ascii_case(&self, other: &str) -> bool {
        self.as_bytes().eq_ignore_ascii_case(other.as_bytes())
    }

    /// Converts this string to its ASCII upper case equivalent in-place.
    ///
    /// ASCII letters 'a' to 'z' are mapped to 'A' to 'Z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To return a new uppercased value without modifying the existing one, use
    /// [`to_ascii_uppercase`].
    ///
    /// [`to_ascii_uppercase`]: #method.to_ascii_uppercase
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    pub fn make_ascii_uppercase(&mut self) {
        let me = unsafe { self.as_bytes_mut() };
        me.make_ascii_uppercase()
    }

    /// Converts this string to its ASCII lower case equivalent in-place.
    ///
    /// ASCII letters 'A' to 'Z' are mapped to 'a' to 'z',
    /// but non-ASCII letters are unchanged.
    ///
    /// To return a new lowercased value without modifying the existing one, use
    /// [`to_ascii_lowercase`].
    ///
    /// [`to_ascii_lowercase`]: #method.to_ascii_lowercase
    #[stable(feature = "ascii_methods_on_intrinsics", since = "1.23.0")]
    pub fn make_ascii_lowercase(&mut self) {
        let me = unsafe { self.as_bytes_mut() };
        me.make_ascii_lowercase()
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
impl Default for &str {
    /// Creates an empty str
    fn default() -> Self { "" }
}

#[stable(feature = "default_mut_str", since = "1.28.0")]
impl Default for &mut str {
    /// Creates an empty mutable str
    fn default() -> Self { unsafe { from_utf8_unchecked_mut(&mut []) } }
}

/// An iterator over the non-whitespace substrings of a string,
/// separated by any amount of whitespace.
///
/// This struct is created by the [`split_whitespace`] method on [`str`].
/// See its documentation for more.
///
/// [`split_whitespace`]: ../../std/primitive.str.html#method.split_whitespace
/// [`str`]: ../../std/primitive.str.html
#[stable(feature = "split_whitespace", since = "1.1.0")]
#[derive(Clone, Debug)]
pub struct SplitWhitespace<'a> {
    inner: Filter<Split<'a, IsWhitespace>, IsNotEmpty>,
}

/// An iterator over the non-ASCII-whitespace substrings of a string,
/// separated by any amount of ASCII whitespace.
///
/// This struct is created by the [`split_ascii_whitespace`] method on [`str`].
/// See its documentation for more.
///
/// [`split_ascii_whitespace`]: ../../std/primitive.str.html#method.split_ascii_whitespace
/// [`str`]: ../../std/primitive.str.html
#[unstable(feature = "split_ascii_whitespace", issue = "48656")]
#[derive(Clone, Debug)]
pub struct SplitAsciiWhitespace<'a> {
    inner: Map<Filter<SliceSplit<'a, u8, IsAsciiWhitespace>, IsNotEmpty>, UnsafeBytesToStr>,
}

#[derive(Clone)]
struct IsWhitespace;

impl FnOnce<(char, )> for IsWhitespace {
    type Output = bool;

    #[inline]
    extern "rust-call" fn call_once(mut self, arg: (char, )) -> bool {
        self.call_mut(arg)
    }
}

impl FnMut<(char, )> for IsWhitespace {
    #[inline]
    extern "rust-call" fn call_mut(&mut self, arg: (char, )) -> bool {
        arg.0.is_whitespace()
    }
}

#[derive(Clone)]
struct IsAsciiWhitespace;

impl<'a> FnOnce<(&'a u8, )> for IsAsciiWhitespace {
    type Output = bool;

    #[inline]
    extern "rust-call" fn call_once(mut self, arg: (&u8, )) -> bool {
        self.call_mut(arg)
    }
}

impl<'a> FnMut<(&'a u8, )> for IsAsciiWhitespace {
    #[inline]
    extern "rust-call" fn call_mut(&mut self, arg: (&u8, )) -> bool {
        arg.0.is_ascii_whitespace()
    }
}

#[derive(Clone)]
struct IsNotEmpty;

impl<'a, 'b> FnOnce<(&'a &'b str, )> for IsNotEmpty {
    type Output = bool;

    #[inline]
    extern "rust-call" fn call_once(mut self, arg: (&'a &'b str, )) -> bool {
        self.call_mut(arg)
    }
}

impl<'a, 'b> FnMut<(&'a &'b str, )> for IsNotEmpty {
    #[inline]
    extern "rust-call" fn call_mut(&mut self, arg: (&'a &'b str, )) -> bool {
        !arg.0.is_empty()
    }
}

impl<'a, 'b> FnOnce<(&'a &'b [u8], )> for IsNotEmpty {
    type Output = bool;

    #[inline]
    extern "rust-call" fn call_once(mut self, arg: (&'a &'b [u8], )) -> bool {
        self.call_mut(arg)
    }
}

impl<'a, 'b> FnMut<(&'a &'b [u8], )> for IsNotEmpty {
    #[inline]
    extern "rust-call" fn call_mut(&mut self, arg: (&'a &'b [u8], )) -> bool {
        !arg.0.is_empty()
    }
}

#[derive(Clone)]
struct UnsafeBytesToStr;

impl<'a> FnOnce<(&'a [u8], )> for UnsafeBytesToStr {
    type Output = &'a str;

    #[inline]
    extern "rust-call" fn call_once(mut self, arg: (&'a [u8], )) -> &'a str {
        self.call_mut(arg)
    }
}

impl<'a> FnMut<(&'a [u8], )> for UnsafeBytesToStr {
    #[inline]
    extern "rust-call" fn call_mut(&mut self, arg: (&'a [u8], )) -> &'a str {
        unsafe { from_utf8_unchecked(arg.0) }
    }
}


#[stable(feature = "split_whitespace", since = "1.1.0")]
impl<'a> Iterator for SplitWhitespace<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[stable(feature = "split_whitespace", since = "1.1.0")]
impl<'a> DoubleEndedIterator for SplitWhitespace<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a str> {
        self.inner.next_back()
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl FusedIterator for SplitWhitespace<'_> {}

#[unstable(feature = "split_ascii_whitespace", issue = "48656")]
impl<'a> Iterator for SplitAsciiWhitespace<'a> {
    type Item = &'a str;

    #[inline]
    fn next(&mut self) -> Option<&'a str> {
        self.inner.next()
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

#[unstable(feature = "split_ascii_whitespace", issue = "48656")]
impl<'a> DoubleEndedIterator for SplitAsciiWhitespace<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<&'a str> {
        self.inner.next_back()
    }
}

#[unstable(feature = "split_ascii_whitespace", issue = "48656")]
impl FusedIterator for SplitAsciiWhitespace<'_> {}

/// An iterator of [`u16`] over the string encoded as UTF-16.
///
/// [`u16`]: ../../std/primitive.u16.html
///
/// This struct is created by the [`encode_utf16`] method on [`str`].
/// See its documentation for more.
///
/// [`encode_utf16`]: ../../std/primitive.str.html#method.encode_utf16
/// [`str`]: ../../std/primitive.str.html
#[derive(Clone)]
#[stable(feature = "encode_utf16", since = "1.8.0")]
pub struct EncodeUtf16<'a> {
    chars: Chars<'a>,
    extra: u16,
}

#[stable(feature = "collection_debug", since = "1.17.0")]
impl fmt::Debug for EncodeUtf16<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.pad("EncodeUtf16 { .. }")
    }
}

#[stable(feature = "encode_utf16", since = "1.8.0")]
impl<'a> Iterator for EncodeUtf16<'a> {
    type Item = u16;

    #[inline]
    fn next(&mut self) -> Option<u16> {
        if self.extra != 0 {
            let tmp = self.extra;
            self.extra = 0;
            return Some(tmp);
        }

        let mut buf = [0; 2];
        self.chars.next().map(|ch| {
            let n = ch.encode_utf16(&mut buf).len();
            if n == 2 {
                self.extra = buf[1];
            }
            buf[0]
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (low, high) = self.chars.size_hint();
        // every char gets either one u16 or two u16,
        // so this iterator is between 1 or 2 times as
        // long as the underlying iterator.
        (low, high.and_then(|n| n.checked_mul(2)))
    }
}

#[stable(feature = "fused", since = "1.26.0")]
impl FusedIterator for EncodeUtf16<'_> {}
